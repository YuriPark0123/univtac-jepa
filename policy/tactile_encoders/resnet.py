import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.ops.misc import FrozenBatchNorm2d

from .base import TactileEncoderBase, resolve_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OriginalTactileEncoder(TactileEncoderBase):
    """
    UniVTAC shared tactile encoder (data/checkpoints/encoder.pth): ResNet18 whose fc maps to 512,
    pre-trained by reconstructing marker / rgb / marked_rgb / depth / pose. Same input handling as
    policy/ACT: one frame, resized to 256x256, values in [0, 1], no mean/std normalisation.
    """
    out_dim = 512
    frame_offsets = (0,)

    def __init__(self, ckpt="data/checkpoints/encoder.pth", image_size=256):
        super().__init__()
        self.image_size = image_size
        self.backbone = torchvision.models.resnet18(num_classes=self.out_dim, norm_layer=FrozenBatchNorm2d)
        ckpt = resolve_path(ckpt)
        assert ckpt is not None and ckpt.exists(), f"tactile checkpoint not found: {ckpt}"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        state = {k: v for k, v in state.items() if not k.endswith("num_batches_tracked")}
        self.backbone.load_state_dict(state, strict=True)

    def forward(self, frames):
        x = frames[:, 0]
        x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        return self.backbone(x)


class ImageNetResNet18Encoder(TactileEncoderBase):
    """ImageNet-pretrained ResNet18 (FrozenBatchNorm, like the ACT vision backbone), global-average-pooled."""
    out_dim = 512
    frame_offsets = (0,)

    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                                          norm_layer=FrozenBatchNorm2d)
        net.fc = nn.Identity()
        self.backbone = net

    def forward(self, frames):
        x = frames[:, 0]
        x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return self.backbone(x)
