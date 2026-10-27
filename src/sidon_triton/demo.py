"""The official Sidon v0.1 inference path, verbatim from
huggingface.co/spaces/sarulab-speech/sidon_demo_beta/app.py (TorchScript
fp32 models + transformers' CPU SeamlessM4TFeatureExtractor). Used as the
speed baseline and as the parity reference for ``SidonPipeline.restore``.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from . import frontend


class DemoSidon:
    def __init__(self, torchscript_dir: Path | None = None, device: str = "cuda",
                 profile: bool = True) -> None:
        from transformers import SeamlessM4TFeatureExtractor

        from .hub import torchscript_dir as resolve

        torchscript_dir = resolve(torchscript_dir)
        suffix = "cuda" if device.startswith("cuda") else "cpu"
        self.device = device
        self.fe = torch.jit.load(
            str(torchscript_dir / f"feature_extractor_{suffix}.pt"),
            map_location=device,
        ).eval()
        self.decoder = torch.jit.load(
            str(torchscript_dir / f"decoder_{suffix}.pt"), map_location=device
        ).eval()
        self.preprocessor = SeamlessM4TFeatureExtractor(
            padding_value=frontend.PAD_VALUE
        )
        # profile: synchronize after every stage for the timing breakdown.
        # Off, the demo's natural overlap stays (CPU fbank of the next chunk
        # runs while the GPU still works on the previous one).
        self.profile = profile
        self.timings: dict[str, float] = {}

    def _tick(self, name: str, started: float) -> float:
        if not self.profile:
            return started
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + (now - started)
        return now

    @torch.inference_mode()
    def restore(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_seconds: int = 96,
        per_channel: bool = False,
    ) -> torch.Tensor:
        """[samples] / [channels, samples] -> 48 kHz restored [samples]
        (per_channel: each channel on its own -> [channels, samples])."""
        waveform = torch.as_tensor(waveform)
        if per_channel and waveform.ndim == 2 and waveform.shape[0] > 1:
            return torch.stack(
                [self.restore(c, sample_rate, chunk_seconds) for c in waveform]
            )
        self.timings = {}
        started = time.perf_counter()
        waveform = torch.as_tensor(waveform)
        target = int(48_000 / sample_rate * waveform.shape[-1])
        wav_16k = frontend.preprocess(waveform, sample_rate).view(1, -1)
        started = self._tick("preprocess", started)
        restoreds = []
        feature_cache = None
        wav_16k = torch.nn.functional.pad(wav_16k, (0, 24000))
        for chunk in wav_16k.view(-1).split(16000 * chunk_seconds):
            inputs = self.preprocessor(
                torch.nn.functional.pad(chunk, (160, 160)).numpy(),
                sampling_rate=16_000,
                return_tensors="pt",
            )
            started = self._tick("fbank_cpu", started)
            feature = self.fe(inputs["input_features"].to(self.device))[
                "last_hidden_state"
            ]
            started = self._tick("encoder", started)
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
            restoreds.append(
                self.decoder(feature.transpose(1, 2)).view(-1)[:-960]
            )
            feature_cache = feature[:, -1:]
            started = self._tick("decoder", started)
        return torch.cat(restoreds, dim=0)[:target]
