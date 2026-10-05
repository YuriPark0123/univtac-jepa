"""
Sparsh tactile encoders (facebook/sparsh-*-base, CC-BY-NC 4.0, research use only).

Preprocessing follows the Sparsh pre-training pipeline (third_party/sparsh, commit fee6a05):
  tactile_ssl/data/vision_tactile.py:_get_tactile_images   frames [t, t-stride, ...], current frame first
  tactile_ssl/data/digit/utils.py:compute_diff              clip((img - bg)/255 + 0.5, 0, 1), uint8
  tactile_ssl/data/digit/utils.py:load_sample_from_buf      rotate 90 deg clockwise if h < w, center crop to h/w = 4/3
  tactile_ssl/data/digit/utils.py:get_resize_transform      Resize((320, 240), antialias) + ToTensor -> [0, 1], no mean/std
  config/default.yaml                                       num_frames 2, frame_stride 5   (mae / dino / ijepa)
  config/experiment/vjepa_vit.yaml                          num_frames 4, frame_stride 2, video (B, C, T, H, W)
The background is a no-contact frame; here it is the first frame of the episode (frame offset "first").
The encoder returns patch tokens (no CLS token); they are mean-pooled into one feature per sensor.
"""
import sys

import torch
from torchvision.transforms import functional as TF

from .base import TactileEncoderBase, UNIVTAC_ROOT, resolve_path

SPARSH_ROOT = UNIVTAC_ROOT / "third_party" / "sparsh"
FIRST = "first"  # frame offset meaning "first frame of the episode" (background)

VARIANTS = {
    # name: weights file, input channels, video frames, register tokens, default frame stride (data frames)
    "mae": dict(weights="data/sparsh/mae_vitbase.safetensors", in_chans=6, num_frames=1, registers=0, stride=5, clip=2),
    "dino": dict(weights="data/sparsh/dino_vitbase.safetensors", in_chans=6, num_frames=1, registers=1, stride=5, clip=2),
    "ijepa": dict(weights="data/sparsh/ijepa_vitbase.safetensors", in_chans=6, num_frames=1, registers=1, stride=5, clip=2),
    "vjepa": dict(weights="data/sparsh/vjepa_vitbase.safetensors", in_chans=3, num_frames=4, registers=1, stride=2, clip=4),
}


class SparshEncoder(TactileEncoderBase):
    out_dim = 768

    def __init__(self, variant="ijepa", weights=None, frame_stride=None, remove_bg=True, img_size=(320, 240)):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.is_video = cfg["num_frames"] > 1
        self.remove_bg = remove_bg
        self.img_size = tuple(img_size)
        self.frame_offsets = self.get_frame_offsets(variant=variant, frame_stride=frame_stride, remove_bg=remove_bg)
        self.num_clip_frames = cfg["clip"]

        if str(SPARSH_ROOT) not in sys.path:
            sys.path.insert(0, str(SPARSH_ROOT))
        from tactile_ssl.model.vision_transformer import vit_base
        from safetensors.torch import load_file

        self.vit = vit_base(
            img_size=self.img_size,
            in_chans=cfg["in_chans"],
            num_frames=cfg["num_frames"],
            tubelet_size=2,
            pos_embed_fn="sinusoidal",
            num_register_tokens=cfg["registers"],
        )
        path = resolve_path(weights or cfg["weights"])
        assert path.exists(), f"Sparsh weights not found: {path}"
        self.vit.load_state_dict(load_file(str(path)), strict=True)

    @classmethod
    def get_frame_offsets(cls, variant="ijepa", frame_stride=None, remove_bg=True, **kwargs):
        cfg = VARIANTS[variant]
        stride = cfg["stride"] if frame_stride is None else frame_stride
        offsets = tuple(-i * stride for i in range(cfg["clip"]))
        return offsets + ((FIRST,) if remove_bg else ())

    def preprocess(self, frames):
        """frames (B, F, 3, H, W) in [0, 1] ordered like frame_offsets -> model input"""
        clip = frames[:, :self.num_clip_frames]
        if self.remove_bg:
            bg = frames[:, -1:]
            clip = torch.floor((clip - bg + 0.5).clamp(0.0, 1.0) * 255.0) / 255.0
        b, t = clip.shape[:2]
        x = clip.flatten(0, 1)
        h, w = x.shape[-2:]
        if h < w:
            x = torch.rot90(x, k=-1, dims=(-2, -1))  # cv2.ROTATE_90_CLOCKWISE
            h, w = w, h
        if h / w != 4 / 3:
            x = TF.center_crop(x, [int(h / (4 / 3)), w])
        x = TF.resize(x, list(self.img_size), antialias=True)
        x = x.view(b, t, *x.shape[1:])
        if self.is_video:
            return x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return x.flatten(1, 2)  # (B, T*C, H, W), channels [t, t-stride]

    def forward(self, frames):
        x = self.preprocess(frames)
        frozen = not any(p.requires_grad for p in self.vit.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            tokens = self.vit.forward_features(x)["x_norm_patchtokens"]  # (B, N, 768)
        return tokens.mean(dim=1)


class CustomJEPAEncoder(SparshEncoder):
    """Sparsh architecture with weights fine-tuned on UniVTAC tactile frames (stage 6). `weights` is required."""

    def __init__(self, weights, variant="ijepa", **kwargs):
        super().__init__(variant=variant, weights=weights, **kwargs)
