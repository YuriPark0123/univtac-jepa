import os
from pathlib import Path

import torch
from torch import nn

# This package lives in the univtac-jepa repo and is linked into <UniVTAC>/policy, so the UniVTAC root
# cannot be derived from __file__. Inside the container it is always /workspace/UniVTAC.
UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))


def resolve_path(path):
    """Relative checkpoint paths are resolved against the UniVTAC root, not the current directory."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else UNIVTAC_ROOT / path


class TactileEncoderBase(nn.Module):
    out_dim: int = None
    # offsets (in recorded data frames) of the frames the encoder needs, current frame first
    frame_offsets: tuple = (0,)

    @classmethod
    def get_frame_offsets(cls, **kwargs):
        """Frame offsets for given constructor kwargs, available before the (heavy) encoder is built."""
        return tuple(cls.frame_offsets)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def freeze_module(module: nn.Module):
    """Stop gradients and pin the module to eval mode (dropout / norm statistics stay fixed)."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    module.train = lambda mode=True: module
    return module
