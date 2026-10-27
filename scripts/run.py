#!/usr/bin/env python3
"""Restore speech files with Sidon-Triton (48 kHz output).

    run.py --engine-dir E --output-dir out/ a.wav b.flac ...
    run.py --engine-dir E --output-dir out/ --input-dir corpus/ [--glob '**/*.flac']
    run.py --engine-dir E --output-dir out/ --list files.txt

Any length, sample rate and channel count; processing follows the
official demo (96 s chunks, 1-frame feature carry-over). --per-channel
restores each channel on its own (2-channel dialogue).
Decoding and encoding run in worker processes overlapped with the GPU
(sidon_triton.stream); --input-dir mirrors its tree under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_triton.runtime import SidonPipeline  # noqa: E402
from sidon_triton.stream import FileStreamer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--glob", default="**/*")
    parser.add_argument("--list", type=Path, default=None,
                        help="text file with one input path per line")
    parser.add_argument("--format", choices=["wav", "flac"], default="wav")
    parser.add_argument("--chunk-seconds", type=float, default=96.0)
    parser.add_argument(
        "--per-channel", action="store_true",
        help="restore each channel separately and keep the channel layout "
        "(2-channel dialogue, one speaker per channel); default downmixes "
        "like the official demo",
    )
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--writers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=32)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("wavs", type=Path, nargs="*")
    args = parser.parse_args()

    audio_suffixes = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
    paths = list(args.wavs)
    if args.input_dir is not None:
        paths += sorted(
            p for p in args.input_dir.glob(args.glob)
            if p.is_file() and p.suffix.lower() in audio_suffixes
        )
    if args.list is not None:
        paths += [
            Path(line.strip()) for line in args.list.read_text().splitlines()
            if line.strip()
        ]
    if not paths:
        parser.error("no input files")

    pipeline = SidonPipeline(args.engine_dir)
    with FileStreamer(pipeline, args.readers, args.writers) as streamer:
        stats = streamer.run(
            paths, args.output_dir, root=args.input_dir, fmt=args.format,
            per_channel=args.per_channel, prefetch=args.prefetch,
            chunk_seconds=args.chunk_seconds, log_every=500,
        )
    summary = {k: v for k, v in stats.items() if k != "errors"}
    summary["errors"] = len(stats["errors"])
    print(json.dumps(summary, indent=2))
    for path, error in list(stats["errors"].items())[:5]:
        print(f"ERROR {path}: {error}", file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
