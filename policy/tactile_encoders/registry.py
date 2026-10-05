from .base import freeze_module
from .resnet import OriginalTactileEncoder, ImageNetResNet18Encoder
from .sparsh import SparshEncoder, CustomJEPAEncoder
from functools import partial

ENCODERS = {
    "original": OriginalTactileEncoder,
    "resnet18_imagenet": ImageNetResNet18Encoder,
    "sparsh_mae": partial(SparshEncoder, variant="mae"),
    "sparsh_dino": partial(SparshEncoder, variant="dino"),
    "sparsh_ijepa": partial(SparshEncoder, variant="ijepa"),
    "sparsh_vjepa": partial(SparshEncoder, variant="vjepa"),
    "custom_jepa": CustomJEPAEncoder,
}


def build_tactile_encoder(name: str, freeze: bool, **kwargs):
    if name not in ENCODERS:
        raise ValueError(f"unknown tactile_encoder '{name}', choose from {sorted(ENCODERS)}")
    encoder = ENCODERS[name](**kwargs)
    if freeze:
        freeze_module(encoder)
    return encoder


def get_frame_offsets(name: str, **kwargs):
    builder = ENCODERS[name]
    if isinstance(builder, partial):
        return builder.func.get_frame_offsets(**{**builder.keywords, **kwargs})
    return builder.get_frame_offsets(**kwargs)
