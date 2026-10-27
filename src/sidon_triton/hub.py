"""Released Sidon v0.1 TorchScript files from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

SIDON_REPO = "sarulab-speech/sidon-v0.1"
# the revision every number in README.md was measured against
SIDON_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
HELP = (
    "directory with the released sidon-v0.1 TorchScript files "
    "(default: download sarulab-speech/sidon-v0.1 from the Hugging Face Hub)"
)


def torchscript_dir(path=None) -> Path:
    """`path` if given, else the Hub snapshot (downloaded once, cached)."""
    if path is not None:
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        SIDON_REPO, revision=SIDON_REVISION, allow_patterns=["*.pt"]
    ))
