"""
Shared tactile encoders for UniVTAC policies (ACT_TacEnc now, other policy heads later).

Every encoder takes raw tactile frames and returns one global feature per sensor:
    forward(frames): frames (B, F, 3, H, W) float in [0, 1], RGB as stored by UniVTAC
                     F = len(encoder.frame_offsets); frames[:, i] is the frame at data offset frame_offsets[i]
                     -> (B, encoder.out_dim)
Encoders do their own resizing / normalisation so callers only pass raw frames.

    build_tactile_encoder(name, freeze, **kwargs)
        name   : original | resnet18_imagenet | sparsh_mae | sparsh_dino | sparsh_ijepa | sparsh_vjepa | custom_jepa
        freeze : True -> no gradients and always in eval mode
"""
from .base import TactileEncoderBase, freeze_module
from .registry import build_tactile_encoder, get_frame_offsets, ENCODERS

__all__ = ["TactileEncoderBase", "freeze_module", "build_tactile_encoder", "get_frame_offsets", "ENCODERS"]
