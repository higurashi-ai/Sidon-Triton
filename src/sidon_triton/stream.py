"""Overlapped file processing: decode / GPU restore / encode in parallel.

    from sidon_triton.runtime import SidonPipeline
    from sidon_triton.stream import FileStreamer
    with FileStreamer(SidonPipeline(engine_dir)) as streamer:
        stats = streamer.run(paths, out_dir)

- decoding runs in a spawned worker-process pool (WAV via the numpy reader,
  anything else via torchaudio/ffmpeg), at most `prefetch` files ahead;
- the calling thread runs ``restore`` (GPU) file after file and converts
  the result to int16 on the GPU (half the bytes to hand over);
- encoding (WAV / FLAC) runs in a second worker-process pool.

Processes rather than threads: torchaudio's decoder and encoder hold the
GIL, so decode threads serialized with the GPU loop (measured: decode
throughput flat from 1 to 16 threads, GPU idle 30% of the wall time).
Per-file failures are collected, not raised.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from . import fileio
from .runtime import OUT_RATE, SidonPipeline


def output_path(path: Path, root: Path | None, out_dir: Path,
                fmt: str) -> Path:
    """mirror the input tree under out_dir when `root` is given."""
    relative = path.relative_to(root) if root is not None else Path(path.name)
    return (out_dir / relative).with_suffix(f".{fmt}")


class FileStreamer:
    def __init__(self, pipeline: SidonPipeline, readers: int = 8,
                 writers: int = 8) -> None:
        self.pipeline = pipeline
        context = mp.get_context("spawn")  # CUDA is initialized: no fork
        self.readers = ProcessPoolExecutor(readers, mp_context=context)
        self.writers = ProcessPoolExecutor(writers, mp_context=context)
        self.max_writes = 4 * writers

    def close(self) -> None:
        self.readers.shutdown()
        self.writers.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(
        self,
        paths: list,
        out_dir: Path,
        root: Path | None = None,
        fmt: str = "wav",
        per_channel: bool = False,
        prefetch: int = 32,
        chunk_seconds: float | None = None,
        log_every: int = 0,
    ) -> dict:
        """Restore every file into out_dir; throughput stats + errors."""
        paths = [Path(p) for p in paths]
        out_dir = Path(out_dir)
        kwargs = (
            {} if chunk_seconds is None else {"chunk_seconds": chunk_seconds}
        )
        errors: dict[str, str] = {}
        decodes: deque = deque()
        writes: deque = deque()
        audio_seconds = 0.0
        gpu_seconds = 0.0
        started = time.perf_counter()
        next_index = 0

        def refill() -> None:
            nonlocal next_index
            while next_index < len(paths) and len(decodes) < prefetch:
                path = paths[next_index]
                decodes.append(
                    (path, self.readers.submit(fileio.decode, str(path)))
                )
                next_index += 1

        def collect(block: bool) -> None:
            while writes and (block or writes[0][1].done()):
                path, future = writes.popleft()
                try:
                    future.result()
                except Exception:  # noqa: BLE001
                    errors[str(path)] = traceback.format_exc(limit=3)

        refill()
        done = 0
        while decodes:
            path, future = decodes.popleft()
            refill()
            try:
                data, rate = future.result()
                tick = time.perf_counter()
                audio = self.pipeline.restore(
                    torch.from_numpy(data), rate, per_channel=per_channel,
                    **kwargs,
                )
                pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16)
                pcm = pcm.cpu().numpy()  # syncs the GPU
                gpu_seconds += time.perf_counter() - tick
                audio_seconds += data.shape[-1] / rate * (
                    data.shape[0] if per_channel else 1
                )
                target = output_path(path, root, out_dir, fmt)
                writes.append((target, self.writers.submit(
                    fileio.encode, str(target), pcm, OUT_RATE, fmt
                )))
            except Exception:  # noqa: BLE001 - keep going, report at end
                errors[str(path)] = traceback.format_exc(limit=3)
            collect(block=False)
            while len(writes) > self.max_writes:  # bound queued output
                path_w, future_w = writes.popleft()
                try:
                    future_w.result()
                except Exception:  # noqa: BLE001
                    errors[str(path_w)] = traceback.format_exc(limit=3)
            done += 1
            if log_every and done % log_every == 0:
                wall = time.perf_counter() - started
                print(f"  {done}/{len(paths)} files, "
                      f"{audio_seconds / max(wall, 1e-9):.0f}x realtime",
                      flush=True)
        collect(block=True)
        wall = time.perf_counter() - started
        return {
            "files": len(paths),
            "audio_seconds": audio_seconds,
            "wall_s": wall,
            "x_realtime": audio_seconds / wall if wall else 0.0,
            "gpu_busy_s": gpu_seconds,
            "gpu_busy_fraction": gpu_seconds / wall if wall else 0.0,
            "errors": errors,
        }


def stream_files(pipeline: SidonPipeline, paths: list, out_dir: Path,
                 readers: int = 8, writers: int = 8, **kwargs) -> dict:
    """One-shot convenience wrapper around :class:`FileStreamer`."""
    with FileStreamer(pipeline, readers, writers) as streamer:
        return streamer.run(paths, out_dir, **kwargs)
