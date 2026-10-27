"""Sidon v0.1 input pipeline, matching the official demo
(``huggingface.co/spaces/sarulab-speech/sidon_demo_beta``):

1. ``preprocess``: peak-normalize to 0.9, mono, 50 Hz high-pass biquad at the
   ORIGINAL rate, sinc resample to 16 kHz (CPU torchaudio, as the demo).
2. ``SeamlessFbank``: GPU fp32 twin of ``transformers``'
   ``SeamlessM4TFeatureExtractor`` (w2v-bert-2.0 preprocessor config) for
   one chunk padded (160, 160): x 2**15, kaldi framing (DC removal,
   pre-emphasis 0.97, povey window, 512-point power spectrum), 80 kaldi
   mel bins, ``log(max(mel, 1.19e-7))``, per-bin normalization with
   ``var(ddof=1) + 1e-7``, an odd frame count padded with 1.0
   (``padding_value`` of the preprocessor config), stride-2 stacking.

The whole per-frame chain DC removal -> pre-emphasis -> window -> rDFT is
linear in the 400-sample frame, so it is precomputed (float64) as one
[400, 514] matrix and applied as a single fp32 GEMM.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16_000
FRAME_SIZE = 400
FRAME_SHIFT = 160
N_FFT = 512
N_BINS = N_FFT // 2 + 1
NUM_MEL = 80
PREEMPH = 0.97
MEL_FLOOR = 1.192092955078125e-07
VAR_EPS = 1e-7
PAD_VALUE = 1.0  # w2v-bert-2.0 preprocessor_config.json "padding_value"
KALDI_SCALE = 2.0**15
EDGE_PAD = 160  # the demo pads every chunk (160, 160) before the extractor

HIGHPASS_HZ = 50.0
PEAK = 0.9


def preprocess(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """[samples] or [channels, samples] at any rate -> [samples16k] fp32.

    Same order as the demo: peak-normalize (over all channels), downmix,
    high-pass at the source rate, resample to 16 kHz.
    """
    import torchaudio.functional as AF

    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = PEAK * (waveform / waveform.abs().max())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = AF.highpass_biquad(waveform, sample_rate, HIGHPASS_HZ)
    if sample_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform[0].contiguous()


_HIGHPASS_IR: dict[tuple[int, str], torch.Tensor] = {}


def _highpass_impulse_response(
    sample_rate: int, device: torch.device
) -> torch.Tensor:
    """float64 impulse response of the demo's biquad, truncated once its
    tail drops below 1e-14 of the peak (a pole radius < 1 makes the IIR
    an absolutely-summable FIR to machine precision)."""
    key = (sample_rate, str(device))
    if key not in _HIGHPASS_IR:
        import torchaudio.functional as AF

        length = 1 << 12
        while True:
            delta = torch.zeros(1, length, dtype=torch.float64)
            delta[0, 0] = 1.0
            response = AF.highpass_biquad(delta, sample_rate, HIGHPASS_HZ)[0]
            tail = response[-length // 4 :].abs().max()
            if tail < 1e-14 * response.abs().max():
                break
            length *= 2
        _HIGHPASS_IR[key] = response.to(device)
    return _HIGHPASS_IR[key]


def _fft_causal_filter(signal: torch.Tensor, response: torch.Tensor):
    """Causal linear convolution y[n] = sum_k h[k] x[n-k], zero history."""
    total = signal.shape[-1] + response.shape[-1] - 1
    size = 1 << (total - 1).bit_length()
    spectrum = torch.fft.rfft(signal, size) * torch.fft.rfft(response, size)
    return torch.fft.irfft(spectrum, size)[..., : signal.shape[-1]]


def preprocess_gpu(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """GPU twin of :func:`preprocess` (same order and kernels).

    The high-pass biquad (sequential IIR in torchaudio.lfilter) runs as an
    fp64 FFT convolution with its truncated impulse response, then the
    lfilter clamp to [-1, 1]; the resampler is the identical torchaudio
    sinc conv on the GPU.
    """
    import torchaudio.functional as AF

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.float()
    waveform = PEAK * (waveform / waveform.abs().max())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    response = _highpass_impulse_response(sample_rate, waveform.device)
    filtered = _fft_causal_filter(waveform.double(), response)
    waveform = filtered.clamp(-1.0, 1.0).float()
    if sample_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform[0].contiguous()


def num_frames(chunk_samples: int) -> int:
    """fbank frames for a chunk of `chunk_samples` (before stride-2)."""
    padded = chunk_samples + 2 * EDGE_PAD
    return 1 + (padded - FRAME_SIZE) // FRAME_SHIFT


def num_features(chunk_samples: int) -> int:
    """Encoder sequence length (50 Hz) for a chunk."""
    return (num_frames(chunk_samples) + 1) // 2


def _povey_window() -> torch.Tensor:
    n = torch.arange(FRAME_SIZE, dtype=torch.float64)
    return (
        0.5 - 0.5 * torch.cos(2 * math.pi * n / (FRAME_SIZE - 1))
    ) ** 0.85


def _frame_transform() -> torch.Tensor:
    """[400, 514] float64: frame -> (real 257, imag 257), x 2**15."""
    eye = torch.eye(FRAME_SIZE, dtype=torch.float64)
    dc = eye - torch.full(
        (FRAME_SIZE, FRAME_SIZE), 1.0 / FRAME_SIZE, dtype=torch.float64
    )
    pre = eye.clone()
    pre[0, 0] -= PREEMPH  # buffer[0] *= 1 - preemphasis
    for row in range(1, FRAME_SIZE):
        pre[row, row - 1] -= PREEMPH
    combined = torch.diag(_povey_window()) @ pre @ dc  # [400, 400]
    k = torch.arange(N_BINS, dtype=torch.float64)[:, None]
    n = torch.arange(FRAME_SIZE, dtype=torch.float64)[None, :]
    angle = 2 * math.pi * k * n / N_FFT
    real = torch.cos(angle) @ combined
    imag = -torch.sin(angle) @ combined
    return (torch.cat([real, imag], dim=0) * KALDI_SCALE).t().contiguous()


def _mel_filters() -> torch.Tensor:
    from transformers.audio_utils import mel_filter_bank

    filters = mel_filter_bank(
        num_frequency_bins=N_BINS,
        num_mel_filters=NUM_MEL,
        min_frequency=20,
        max_frequency=SAMPLE_RATE // 2,
        sampling_rate=SAMPLE_RATE,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    )
    return torch.from_numpy(np.asarray(filters, dtype=np.float64))  # [257,80]


class SeamlessFbank(nn.Module):
    """chunk [B, samples] (16 kHz, unpadded) -> features [B, L, 160] fp32.

    Must run in true fp32 (TF32 off): quiet-bin log-mel is chaotic under
    TF32 GEMM noise and per-bin normalization amplifies it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frame_weight", _frame_transform().float())
        self.register_buffer("mel_weight", _mel_filters().float())

    def forward(
        self, chunk: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """valid (int [B], optional): true samples per row when `chunk` is
        zero-padded to a bucket length (a multiple of 320 samples). The
        result then equals the exact-length extractor on the valid rows:
        statistics over valid frames only, the odd-count 1.0 pad frame at
        its exact position, zeros after it."""
        if valid is not None:
            return self._masked(chunk, valid)
        padded = torch.nn.functional.pad(chunk, (EDGE_PAD, EDGE_PAD))
        frames = padded.unfold(-1, FRAME_SIZE, FRAME_SHIFT)  # [B, F, 400]
        spectrum = torch.matmul(frames, self.frame_weight)  # [B, F, 514]
        real, imag = spectrum[..., :N_BINS], spectrum[..., N_BINS:]
        power = real * real + imag * imag
        log_mel = torch.matmul(power, self.mel_weight).clamp_min(
            MEL_FLOOR
        ).log()  # [B, F, 80]
        count = log_mel.shape[1]
        mean = log_mel.mean(dim=1, keepdim=True)
        centered = log_mel - mean
        var = centered.square().sum(dim=1, keepdim=True) / (count - 1)
        normed = centered / torch.sqrt(var + VAR_EPS)
        if count % 2:
            normed = torch.cat(
                [normed, normed.new_full((normed.shape[0], 1, NUM_MEL), PAD_VALUE)],
                dim=1,
            )
        batch = normed.shape[0]
        return normed.reshape(batch, normed.shape[1] // 2, 2 * NUM_MEL)

    def _masked(self, chunk: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        assert chunk.shape[-1] % (2 * FRAME_SHIFT) == 0, "bucket % 320 != 0"
        padded = torch.nn.functional.pad(chunk, (EDGE_PAD, EDGE_PAD))
        frames = padded.unfold(-1, FRAME_SIZE, FRAME_SHIFT)  # [B, Fb, 400]
        spectrum = torch.matmul(frames, self.frame_weight)
        real, imag = spectrum[..., :N_BINS], spectrum[..., N_BINS:]
        log_mel = torch.matmul(real * real + imag * imag, self.mel_weight)
        log_mel = log_mel.clamp_min(MEL_FLOOR).log()  # [B, Fb, 80]
        count = log_mel.shape[1]  # Fb, even for 320-multiples
        # frames of the exact-length extractor: 1 + (n + 320 - 400) // 160
        true_frames = (valid.to(torch.long) - 80) // FRAME_SHIFT + 1  # [B]
        index = torch.arange(count, device=chunk.device)[None, :]
        mask = (index < true_frames[:, None]).to(log_mel.dtype)[..., None]
        n = true_frames.to(log_mel.dtype)[:, None, None]
        mean = (log_mel * mask).sum(dim=1, keepdim=True) / n
        centered = (log_mel - mean) * mask
        var = centered.square().sum(dim=1, keepdim=True) / (n - 1)
        normed = centered / torch.sqrt(var + VAR_EPS)
        pad_row = (index == true_frames[:, None]) & (true_frames[:, None] % 2 == 1)
        normed = torch.where(
            pad_row[..., None], torch.full_like(normed, PAD_VALUE), normed
        )
        return normed.reshape(chunk.shape[0], count // 2, 2 * NUM_MEL)


def valid_features(samples: torch.Tensor) -> torch.Tensor:
    """Encoder frames (50 Hz) of chunks with `samples` true samples."""
    true_frames = (samples.to(torch.long) - 80) // FRAME_SHIFT + 1
    return (true_frames + 1) // 2


def reference_features(chunk: torch.Tensor) -> torch.Tensor:
    """The demo's CPU extractor (transformers) on one chunk -> [1, L, 160]."""
    from transformers import SeamlessM4TFeatureExtractor

    extractor = SeamlessM4TFeatureExtractor(padding_value=PAD_VALUE)
    padded = torch.nn.functional.pad(chunk.cpu().float(), (EDGE_PAD, EDGE_PAD))
    return extractor(
        padded.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )["input_features"]
