"""Audio file decode / encode for worker processes (no torch-CUDA or
TensorRT imports: these run in a spawned process pool)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_io import read_wav, write_wav_pcm16


def decode(path: str) -> tuple[np.ndarray, int]:
    """-> (float32 [channels, samples], rate)"""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return read_wav(path)
        except ValueError:
            pass  # unusual WAV flavour: fall back to ffmpeg
    import torchaudio

    waveform, rate = torchaudio.load(str(path))
    return waveform.numpy(), int(rate)


def encode(path: str, pcm16: np.ndarray, rate: int, fmt: str) -> None:
    """int16 [samples] or [channels, samples] -> file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "wav":
        write_wav_pcm16(path, pcm16, rate)
        return
    import torch
    import torchaudio

    audio = np.atleast_2d(pcm16).astype(np.float32) / 32767.0
    torchaudio.save(str(path), torch.from_numpy(audio), rate, format=fmt,
                    bits_per_sample=16)
