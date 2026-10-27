"""Runtime: torch fp32 fbank -> encoder (Triton kernels, or a TensorRT
engine) -> decoder head (Triton GEMM kernels, or a TensorRT engine) ->
decoder tail (Triton kernels, or the fused CUDA kernels).

``SidonPipeline.restore`` reproduces the official demo's long-input
procedure exactly (sidon_demo_beta/app.py):

- preprocess (peak 0.9, 50 Hz high-pass, 16 kHz), append 1.5 s of zeros;
- split into ``chunk_seconds`` (96 s) chunks; every chunk is featurized
  on its own (per-chunk normalization);
- each chunk after the first is decoded with the previous chunk's last
  feature frame prepended; the last 960 output samples of every chunk are
  dropped; the concatenation is trimmed to the 48 kHz input length.

Bucketed CUDA graphs: a chunk of n samples runs zero-padded to the next
bucket (1 s steps up to 32 s, 4 s steps above) with its true length passed
alongside. The fbank statistics, the encoder attention (key mask) and every
decoder conv (re-zeroing past the valid length) ignore the padding, so the
valid output equals an exact-length run while all chunks of a bucket share
one captured graph (front-end + engines + CUDA tail): at most ~48 graphs
for any mix of input lengths.

(Batching several short clips per engine call was measured and dropped:
no throughput gain on H100 — batch 1 already saturates the GPU — for
~20 GB of extra activation memory.)

Enqueue always happens on a non-default stream (FlexRel plugin race rule).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch

from . import (
    cuda_tail,
    flexrel_triton,
    frontend,
    snake_triton,
    triton_encoder,
    triton_head,
    triton_tail,
)
from .engines import TRTStage, deserialize
from .export_onnx import head_mask_lengths, head_masks

IN_RATE, OUT_RATE = 16_000, 48_000
HOP = 960  # output samples per 50 Hz feature frame
TRIM = 26  # DAC transposed-conv length loss: out = 960 * L - 26
SAMPLES_PER_FRAME = 320
MAX_SECONDS = 96  # long engine set default maximum (the demo chunk)
MAX_SEQ = MAX_SECONDS * 50
MIN_SEQ = 2
DEMO_CHUNK_SECONDS = 96
DEMO_TAIL_PAD = 24_000  # 1.5 s at 16 kHz
FINE_BUCKET_SECONDS = 1.0  # bucket step up to FINE_LIMIT_SECONDS
FINE_LIMIT_SECONDS = 32.0
COARSE_BUCKET_SECONDS = 4.0

ENGINE_FILES = {"encoder": "encoder.plan", "decoder": "decoder.plan"}


def max_chunk_samples(seconds: float) -> int:
    return int(seconds * IN_RATE)


def _profile_max(engine, name: str) -> tuple:
    return tuple(engine.get_tensor_profile_shape(name, 0)[2])


class EngineSet:
    """Encoder + decoder-head engines (+ CUDA tail) with flat max-size
    buffers re-viewed per (batch, length): TensorRT needs contiguous
    [B, L, ...] tensors at fixed addresses."""

    def __init__(self, engine_dir: Path, tail,
                 front: frontend.SeamlessFbank,
                 head: triton_head.TritonDecoderHead | None = None,
                 encoder: tuple | None = None) -> None:
        """tail: CudaDecoderTail / TritonDecoderTail (None: TRT-only
        decoder); encoder: (TritonEncoder, max_batch, max_seconds) replaces
        the encoder engine; head: TritonDecoderHead replaces the head
        engine (required by the Triton tail, which takes its Snake'd
        output too)."""
        skip = set()
        if head is not None:
            skip.add("decoder")
        if encoder is not None:
            skip.add("encoder")
        engines = {
            tag: deserialize(engine_dir / name)
            for tag, name in ENGINE_FILES.items() if tag not in skip
        }
        if encoder is None:
            names = {
                engines["encoder"].get_tensor_name(i)
                for i in range(engines["encoder"].num_io_tensors)
            }
            assert "lens" in names, (
                f"{engine_dir}: engines predate bucketed execution (no "
                "`lens` input); rebuild with scripts/build_engines.py"
            )
        self.tail = tail
        self.front = front
        self.head = head
        if head is None:
            decoder = engines["decoder"]
            io = [decoder.get_tensor_name(i)
                  for i in range(decoder.num_io_tensors)]
            # decoder engine: CUDA-tail head (-> z) or TRT-only decoder
            self.trt_decoder = "audio" in io
            self.mask_names = [n for n in io if n.startswith("mask_")]
            self.from_block = len(self.mask_names) - 2
        else:
            self.trt_decoder = False
            self.mask_names = []
            self.from_block = head.from_block
        assert self.trt_decoder or tail is not None, "head needs the tail"
        self.triton_tail = isinstance(tail, triton_tail.TritonDecoderTail)
        assert head is not None or not self.triton_tail, (
            "the Triton tail needs the Triton head"
        )
        if encoder is None:
            self.encoder = None
            self.max_batch, self.max_seq, _ = _profile_max(
                engines["encoder"], "features"
            )
        else:
            self.encoder, self.max_batch, max_seconds = encoder
            self.max_seq = int(max_seconds * 50)
        device = torch.device("cuda")
        b, s = self.max_batch, self.max_seq
        self.max_samples = s * SAMPLES_PER_FRAME
        self.wave = torch.zeros(b * self.max_samples, device=device)
        self.wave_valid = torch.zeros(b, dtype=torch.int32, device=device)
        self.features = torch.zeros(b * s * 160, device=device)
        self.enc_lens = torch.zeros(b, dtype=torch.int32, device=device)
        self.dec_input = torch.zeros(b * (s + 1) * 1024, device=device)
        self.dec_lens = torch.zeros(b, dtype=torch.int32, device=device)
        # head masks (1 = valid) at its four resolutions, flat max-size
        self.mask_max = head_mask_lengths(s + 1, self.from_block)
        self.masks = [
            torch.zeros(b * length, device=device) for length in self.mask_max
        ]
        if encoder is None:
            self.encoder_stage = TRTStage(
                engines["encoder"],
                {
                    "features": self.features.view(b, s, 160),
                    "lens": self.enc_lens,
                },
            )
            self.hidden = self.encoder_stage.outputs["hidden"].view(-1)
        else:
            self.encoder_stage = None
            self.hidden = torch.zeros(b * s * 1024, device=device)
        self.decoder_stage = None if head is not None else TRTStage(
            engines["decoder"],
            {
                "hidden": self.dec_input.view(b, s + 1, 1024),
                **{
                    name: mask.view(b, length)
                    for name, mask, length in zip(
                        self.mask_names, self.masks, self.mask_max
                    )
                },
            },
        )
        if head is not None:
            self.z = None  # the Triton head allocates z (graph pool)
            self.audio = torch.zeros(
                b * (HOP * (s + 1) - TRIM), device=device
            )
        elif self.trt_decoder:
            self.z = None
            self.audio = self.decoder_stage.outputs["audio"].view(-1)
        else:
            self.z = self.decoder_stage.outputs["z"].view(-1)
            self.audio = torch.zeros(
                b * (HOP * (s + 1) - TRIM), device=device
            )
        self._shape = None

    # -- per-shape views ----------------------------------------------------

    def wave_view(self, batch: int, samples: int) -> torch.Tensor:
        return self.wave[: batch * samples].view(batch, samples)

    def set_shapes(self, batch: int, seq: int, dec_seq: int) -> None:
        if self._shape == (batch, seq, dec_seq):
            return
        assert 1 <= batch <= self.max_batch, batch
        assert MIN_SEQ <= seq <= self.max_seq, seq
        if self.encoder_stage is not None:
            enc = self.encoder_stage.context
            enc.set_input_shape("features", (batch, seq, 160))
            enc.set_input_shape("lens", (batch,))
        if self.decoder_stage is not None:
            dec = self.decoder_stage.context
            dec.set_input_shape("hidden", (batch, dec_seq, 1024))
            for name, length in zip(
                self.mask_names, head_mask_lengths(dec_seq, self.from_block)
            ):
                dec.set_input_shape(name, (batch, length))
        self._shape = (batch, seq, dec_seq)

    def enqueue(self, batch: int, samples: int, cached: bool) -> None:
        """One bucket-padded chunk batch on the current stream: wave rows
        [:batch, :samples] and wave_valid[:batch] already filled. With
        `cached` (batch 1 only) dec_input row 0 holds the carried frame."""
        stream = torch.cuda.current_stream()
        seq = frontend.num_features(samples)
        dec_seq = seq + int(cached)
        self.set_shapes(batch, seq, dec_seq)
        valid = self.wave_valid[:batch]
        features = self.front(self.wave_view(batch, samples), valid)
        self.features[: batch * seq * 160].view(batch, seq, 160).copy_(
            features
        )
        frames = frontend.valid_features(valid)
        self.enc_lens[:batch].copy_(frames)
        hidden = self.run_encoder(batch, seq, stream)
        dec_input = self.dec_input[: batch * dec_seq * 1024].view(
            batch, dec_seq, 1024
        )
        if cached:
            dec_input[:, 1:].copy_(hidden)
        else:
            dec_input.copy_(hidden)
        dec_frames = frames + int(cached)
        self.dec_lens[:batch].copy_(dec_frames)
        z = self.run_head(dec_input, dec_frames, stream)
        if z is None:  # TRT-only decoder: audio written into self.audio
            return
        audio = self.run_tail(z, dec_frames)  # [B, 1, HOP * dec_seq - TRIM]
        self.audio[: audio.numel()].copy_(audio.view(-1))

    def run_encoder(self, batch: int, seq: int, stream) -> torch.Tensor:
        """features[:batch, :seq] (valid frames enc_lens) -> hidden
        [B, L, 1024] fp32 (a view of self.hidden)."""
        hidden = self.hidden[: batch * seq * 1024].view(batch, seq, 1024)
        if self.encoder is None:
            self.encoder_stage.enqueue(stream)
        else:
            features = self.features[: batch * seq * 160].view(
                batch, seq, 160)
            self.encoder(features, self.enc_lens[:batch], out=hidden)
        return hidden

    def run_head(self, dec_input: torch.Tensor, frames, stream):
        """decoder head on dec_input [B, L, 1024] (valid `frames`) -> the
        tail input: z [B, T, C] time-major ((z, snake1(z)) for the Triton
        tail; None: TRT-only decoder)."""
        if self.head is not None:
            return self.head(dec_input, frames, with_snake=self.triton_tail)
        batch, dec_seq, _ = dec_input.shape
        self._fill_masks(batch, dec_seq, frames)
        self.decoder_stage.enqueue(stream)
        if self.trt_decoder:
            return None
        length = cuda_tail.head_samples(dec_seq)
        return self.z[: batch * length * cuda_tail.TAIL_CHANNELS].view(
            batch, length, cuda_tail.TAIL_CHANNELS
        )

    def run_tail(self, z, frames) -> torch.Tensor:
        """run_head output -> audio [B, 1, HOP * L - TRIM]."""
        if self.triton_tail:
            return self.tail(*z, frames)
        return self.tail(z, frames)

    def _fill_masks(self, batch: int, dec_seq: int, frames) -> None:
        lengths = head_mask_lengths(dec_seq, self.from_block)
        for buffer, mask in zip(
            self.masks, head_masks(frames, lengths, self.from_block)
        ):
            buffer[: mask.numel()].view_as(mask).copy_(mask)

    def audio_view(self, batch: int, samples: int, cached: bool):
        dec_seq = frontend.num_features(samples) + int(cached)
        return self.audio[: batch * (HOP * dec_seq - TRIM)].view(batch, -1)

    def last_hidden(self, seq: int, frame: int) -> torch.Tensor:
        """hidden row `frame` of batch item 0 for a [1, seq] run."""
        return self.hidden[frame * 1024 : (frame + 1) * 1024]

    # -- non-graph helpers (tools) ------------------------------------------

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """exact-length features [B, L, 160] -> hidden [B, L, 1024]."""
        batch, seq, _ = features.shape
        self.set_shapes(batch, seq, seq)
        self.features[: features.numel()].view_as(features).copy_(features)
        self.enc_lens[:batch].fill_(seq)
        return self.run_encoder(
            batch, seq, torch.cuda.current_stream()
        ).clone()

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """exact-length hidden [B, L, 1024] -> audio [B, 960 L - 26]."""
        batch, seq, _ = hidden.shape
        self.set_shapes(batch, seq, seq)
        dec_input = self.dec_input[: hidden.numel()].view_as(hidden)
        dec_input.copy_(hidden)
        self.dec_lens[:batch].fill_(seq)
        z = self.run_head(dec_input, self.dec_lens[:batch],
                          torch.cuda.current_stream())
        if z is None:
            audio = self.audio[: batch * (HOP * seq - TRIM)]
            return audio.view(batch, -1).clone()
        return self.run_tail(z, None).view(batch, -1).clone()


class SidonPipeline:
    def __init__(
        self,
        engine_dir: Path,
        use_graph: bool = True,
        max_graphs: int = 128,
    ) -> None:
        assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0", (
            "engines are built with NVIDIA_TF32_OVERRIDE=0 and must run "
            "under the same setting (it also keeps the fbank GEMM fp32)"
        )
        engine_dir = Path(engine_dir)
        enc_path = engine_dir / triton_encoder.ENCODER_FILE
        head_path = engine_dir / triton_head.HEAD_FILE
        self.trt_free = enc_path.exists() and head_path.exists()
        if not self.trt_free:  # TensorRT is only imported when needed
            flexrel_triton.register_trt_plugin_qr_dyn()
            snake_triton.register_trt_plugin()
        # decoder tail: Triton (decoder_tail_triton.pt) or CUDA
        # (decoder_tail.pt); TRT-only decoders have neither. A Triton head
        # (decoder_head.pt) replaces the head engine.
        tail_path = engine_dir / cuda_tail.TAIL_FILE
        ttail_path = engine_dir / triton_tail.TAIL_FILE
        if ttail_path.exists():
            self.tail = triton_tail.load_tail(ttail_path)
        elif tail_path.exists():
            self.tail = cuda_tail.load_tail(tail_path)
        else:
            self.tail = None
        self.head = (
            triton_head.load_head(head_path) if head_path.exists() else None
        )
        self.encoder = (
            triton_encoder.load_encoder(enc_path) if enc_path.exists()
            else None
        )
        self.front = frontend.SeamlessFbank().cuda().eval()
        self.long = EngineSet(engine_dir, self.tail, self.front, self.head,
                              self.encoder)
        self.use_graph = use_graph
        self.max_graphs = max_graphs
        self.compute = torch.cuda.Stream()
        self._graphs: OrderedDict = OrderedDict()
        self._pool = None

    # ---- bucketed graph execution -----------------------------------------

    def _bucketed(self, samples: int) -> int:
        """bucket length (multiple of 320 samples) for a chunk."""
        fine = int(FINE_BUCKET_SECONDS * IN_RATE)
        step = fine if samples <= FINE_LIMIT_SECONDS * IN_RATE else int(
            COARSE_BUCKET_SECONDS * IN_RATE
        )
        return min(self.long.max_samples, -(-samples // step) * step)

    def _run(self, engines: EngineSet, batch: int, samples: int,
             cached: bool) -> None:
        """enqueue (or replay the graph of) one bucket on self.compute."""
        if not self.use_graph:
            engines.enqueue(batch, samples, cached)
            return
        key = (id(engines), batch, samples, cached)
        graph = self._graphs.get(key)
        if graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                engines.enqueue(batch, samples, cached)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # one memory pool for every graph: the CUDA tail allocates
            # inside the graphs, which only ever replay one at a time
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._pool):
                engines.enqueue(batch, samples, cached)
            self._graphs[key] = graph
            while len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        graph.replay()

    def run_chunk(self, samples: int, cached: bool) -> int:
        """Long-set chunk already in wave rows [0, :samples] (true length,
        zero after it) on the CURRENT stream -> valid decoder frames."""
        engines = self.long
        bucket = self._bucketed(samples)
        engines.wave[samples:bucket].zero_()
        engines.wave_valid[:1].fill_(samples)
        self._run(engines, 1, bucket, cached)
        return frontend.num_features(samples) + int(cached)

    # ---- demo-parity restoration -------------------------------------------

    @torch.inference_mode()
    def restore_16k(
        self,
        wav16: torch.Tensor,
        out_samples: int,
        chunk_seconds: float = DEMO_CHUNK_SECONDS,
    ) -> torch.Tensor:
        """preprocessed 16 kHz mono -> 48 kHz restored [out_samples] (GPU)."""
        engines = self.long
        chunk = max_chunk_samples(chunk_seconds)
        assert chunk <= engines.max_samples, (
            f"chunk_seconds={chunk_seconds} exceeds the {MAX_SECONDS} s "
            "engine profile"
        )
        device = engines.wave.device
        pieces = []
        last_frames = 0
        with torch.cuda.stream(self.compute):
            wav16 = torch.nn.functional.pad(
                wav16.to(device, torch.float32).view(-1), (0, DEMO_TAIL_PAD)
            )
            for index, start in enumerate(range(0, wav16.numel(), chunk)):
                piece = wav16[start : start + chunk]
                samples = piece.numel()
                assert frontend.num_features(samples) >= MIN_SEQ, (
                    f"final chunk too short ({samples} samples)"
                )
                cached = index > 0
                if cached:  # previous chunk's last valid hidden frame
                    engines.dec_input[:1024].copy_(
                        engines.last_hidden(0, last_frames - 1)
                    )
                engines.wave[:samples].copy_(piece)
                dec_frames = self.run_chunk(samples, cached)
                last_frames = dec_frames - int(cached)
                produced = HOP * dec_frames - TRIM
                bucket = self._bucketed(samples)
                audio = engines.audio_view(1, bucket, cached)[0]
                pieces.append(audio[: produced - HOP].clone())
        torch.cuda.synchronize()
        return torch.cat(pieces)[:out_samples]

    def _preprocess(self, waveform: torch.Tensor, sample_rate: int,
                    gpu_preprocess: bool = True) -> torch.Tensor:
        if gpu_preprocess:
            with torch.cuda.stream(self.compute):
                return frontend.preprocess_gpu(
                    waveform.to(self.long.wave.device, non_blocking=True),
                    sample_rate,
                )
        return frontend.preprocess(waveform, sample_rate)

    def restore(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_seconds: float = DEMO_CHUNK_SECONDS,
        gpu_preprocess: bool = True,
        per_channel: bool = False,
    ) -> torch.Tensor:
        """[samples] / [channels, samples] at any rate -> 48 kHz [samples].

        gpu_preprocess runs normalize / high-pass / resample on the GPU
        (frontend.preprocess_gpu); False uses the demo's CPU torchaudio path.
        per_channel restores every channel on its own (e.g. 2-channel
        dialogue, one speaker per channel) -> [channels, samples];
        otherwise channels are downmixed as in the official demo.
        """
        waveform = torch.as_tensor(waveform)
        if per_channel and waveform.ndim == 2 and waveform.shape[0] > 1:
            return torch.stack(
                [
                    self.restore(channel, sample_rate, chunk_seconds,
                                 gpu_preprocess)
                    for channel in waveform
                ]
            )
        out_samples = int(OUT_RATE / sample_rate * waveform.shape[-1])
        wav16 = self._preprocess(waveform, sample_rate, gpu_preprocess)
        return self.restore_16k(wav16, out_samples, chunk_seconds)

    # ---- sliding-window restoration ----------------------------------------

    @torch.inference_mode()
    def restore_sliding_16k(
        self,
        wav16: torch.Tensor,
        out_samples: int,
        window_seconds: float = 20.0,
        overlap_seconds: float = 1.0,
    ) -> torch.Tensor:
        """Independent fixed-length windows (Sidon's 20 s training length)
        overlapping by `overlap_seconds`, cross-faded with a raised cosine.

        Each window is featurized on its own (own fbank normalization), run
        through encoder + decoder without frame carry-over, and placed at
        3x its 16 kHz offset in the 48 kHz output. The last window is
        right-aligned so every window has the full length (inputs shorter
        than one window run as a single window). Same 1.5 s tail pad as
        the demo path.
        """
        engines = self.long
        device = engines.wave.device
        window = int(window_seconds * IN_RATE)
        window -= window % (2 * frontend.FRAME_SHIFT)  # whole 50 Hz frames
        hop = window - int(overlap_seconds * IN_RATE)
        assert 0 < hop <= window and window <= engines.max_samples
        ratio = OUT_RATE // IN_RATE
        with torch.cuda.stream(self.compute):
            wav16 = torch.nn.functional.pad(
                wav16.to(device, torch.float32).view(-1), (0, DEMO_TAIL_PAD)
            )
            total = wav16.numel()
            if total <= window:
                starts = [0]
            else:
                starts = list(range(0, total - window, hop)) + [total - window]
            out = torch.zeros(ratio * total + HOP, device=device)
            weight = torch.zeros_like(out)
            produced_prev = 0
            for index, start in enumerate(starts):
                piece = wav16[start : start + window]
                samples = piece.numel()
                engines.wave[:samples].copy_(piece)
                dec_frames = self.run_chunk(samples, cached=False)
                produced = HOP * dec_frames - TRIM
                bucket = self._bucketed(samples)
                audio = engines.audio_view(1, bucket, False)[0, :produced]
                ramp = torch.ones(produced, device=device)
                o0 = ratio * start
                if index > 0:  # fade in over the overlap with the previous
                    prev_end = ratio * starts[index - 1] + produced_prev
                    fade = max(0, min(prev_end - o0, produced))
                    if fade > 0:
                        t = torch.arange(fade, device=device) / fade
                        ramp[:fade] = 0.5 - 0.5 * torch.cos(torch.pi * t)
                if index + 1 < len(starts):  # fade out into the next
                    next_o0 = ratio * starts[index + 1]
                    fade = max(0, min(o0 + produced - next_o0, produced))
                    if fade > 0:
                        t = torch.arange(fade, device=device) / fade
                        ramp[produced - fade :] = 0.5 + 0.5 * torch.cos(
                            torch.pi * t
                        )
                out[o0 : o0 + produced] += audio * ramp
                weight[o0 : o0 + produced] += ramp
                produced_prev = produced
            result = out / weight.clamp_min(1e-6)
        torch.cuda.synchronize()
        return result[:out_samples].clone()

    def restore_sliding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        window_seconds: float = 20.0,
        overlap_seconds: float = 1.0,
        per_channel: bool = False,
    ) -> torch.Tensor:
        """Sliding-window twin of :meth:`restore` (same preprocessing)."""
        waveform = torch.as_tensor(waveform)
        if per_channel and waveform.ndim == 2 and waveform.shape[0] > 1:
            return torch.stack(
                [
                    self.restore_sliding(c, sample_rate, window_seconds,
                                         overlap_seconds)
                    for c in waveform
                ]
            )
        out_samples = int(OUT_RATE / sample_rate * waveform.shape[-1])
        wav16 = self._preprocess(waveform, sample_rate)
        return self.restore_sliding_16k(
            wav16, out_samples, window_seconds, overlap_seconds
        )
