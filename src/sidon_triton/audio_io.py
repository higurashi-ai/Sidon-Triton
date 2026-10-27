"""Dependency-free WAV I/O (numpy): PCM 16/24/32-bit and IEEE float32/64,
including WAVE_FORMAT_EXTENSIBLE headers."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """-> (float32 [channels, samples] in [-1, 1], sample_rate)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    offset = 12
    fmt = None
    payload = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            payload = body
            break
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"{path}: missing fmt/data chunk")
    tag, channels, rate = struct.unpack("<HHI", fmt[:8])
    bits = struct.unpack("<H", fmt[14:16])[0]
    if tag == _EXTENSIBLE:
        tag = struct.unpack("<H", fmt[24:26])[0]
    width = bits // 8
    usable = len(payload) // (width * channels) * width * channels
    payload = payload[:usable]
    if tag == _FLOAT:
        dtype = {4: "<f4", 8: "<f8"}[width]
        samples = np.frombuffer(payload, dtype=dtype).astype(np.float32)
    elif tag == _PCM:
        if width == 2:
            samples = np.frombuffer(payload, "<i2").astype(np.float32) / 32768.0
        elif width == 4:
            samples = (
                np.frombuffer(payload, "<i4").astype(np.float64) / 2**31
            ).astype(np.float32)
        elif width == 3:
            raw = np.frombuffer(payload, np.uint8).reshape(-1, 3)
            ints = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            samples = (ints / float(1 << 23)).astype(np.float32)
        else:
            raise ValueError(f"{path}: unsupported PCM width {bits}")
    else:
        raise ValueError(f"{path}: unsupported format tag {tag}")
    return samples.reshape(-1, channels).T.copy(), int(rate)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """float [-1, 1], [samples] or [channels, samples] -> 16-bit PCM WAV."""
    audio = np.asarray(audio)
    write_wav_pcm16(path, (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2"),
                    rate)


def write_wav_pcm16(path: Path, pcm: np.ndarray, rate: int) -> None:
    """int16 [samples] or [channels, samples] -> 16-bit PCM WAV."""
    pcm = np.asarray(pcm, dtype="<i2")
    if pcm.ndim == 1:
        pcm = pcm[None]
    channels = pcm.shape[0]
    data = np.ascontiguousarray(pcm.T)  # interleave
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data.nbytes, b"WAVE", b"fmt ", 16, _PCM, channels,
        rate, rate * 2 * channels, 2 * channels, 16, b"data", data.nbytes,
    )
    Path(path).write_bytes(header + data.tobytes())
