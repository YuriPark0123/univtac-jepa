"""
1) Sparsh preprocessing in tactile_encoders == the official Sparsh pipeline (third_party/sparsh) on a real isaac45 frame
2) all encoders load strictly and return finite features of the documented size
3) the 'original' encoder holds exactly the weights of data/checkpoints/encoder.pth
"""
import os
import sys

import cv2
import h5py
import numpy as np
import torch

ROOT = os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC")
sys.path[:0] = [f"{ROOT}/policy", f"{ROOT}/third_party/sparsh"]
from tactile_encoders import build_tactile_encoder, get_frame_offsets  # noqa: E402
from tactile_ssl.data.digit.utils import load_sample_from_buf, get_resize_transform  # noqa: E402

DEV = "cuda"
OUT_DIM = {"original": 512, "resnet18_imagenet": 512, "sparsh_mae": 768, "sparsh_dino": 768,
           "sparsh_ijepa": 768, "sparsh_vjepa": 768}

f = h5py.File(f"{ROOT}/data/isaac45/grasp_classify/hdf5/0.hdf5", "r")
key = "tactile/left_gsmini/rgb_marker"
imgs = {i: cv2.imdecode(np.frombuffer(f[key][i], np.uint8), cv2.IMREAD_COLOR) for i in range(0, 11)}
t = 10
to_t = lambda im: torch.from_numpy(im).permute(2, 0, 1).float() / 255

# 1) preprocessing vs Sparsh repo (current frame, frame t-5, background = first frame)
ref = torch.cat([get_resize_transform((320, 240))(load_sample_from_buf(imgs[k], imgs[0])) for k in (t, t - 5)], 0)
frames = torch.stack([to_t(imgs[t]), to_t(imgs[t - 5]), to_t(imgs[0])]).unsqueeze(0).to(DEV)
mine = build_tactile_encoder("sparsh_ijepa", freeze=True).to(DEV).preprocess(frames)[0].cpu()
err = (mine - ref).abs().max().item()
assert mine.shape == ref.shape == (6, 320, 240) and err < 1e-6, (mine.shape, err)
print(f"[ok] sparsh preprocessing matches official pipeline (max diff {err:.1e})")

# 2) every encoder
for name, dim in OUT_DIM.items():
    offs = get_frame_offsets(name)
    pick = {o: imgs[0] if o == "first" else imgs[t + o] for o in offs}
    x = torch.stack([to_t(pick[o]) for o in offs]).unsqueeze(0).repeat(2, 1, 1, 1, 1).to(DEV)
    enc = build_tactile_encoder(name, freeze=True).to(DEV)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, dim) and torch.isfinite(out).all(), (name, out.shape)
    assert not any(p.requires_grad for p in enc.parameters()), name
    print(f"[ok] {name:18s} offsets={offs} -> {tuple(out.shape)}")
    del enc

# 3) original == encoder.pth
enc = build_tactile_encoder("original", freeze=False)
ref_sd = torch.load(f"{ROOT}/data/checkpoints/encoder.pth", map_location="cpu", weights_only=True)
diff = max((v - ref_sd["backbone." + k]).abs().max().item()
           for k, v in enc.backbone.state_dict().items() if "backbone." + k in ref_sd)
assert diff == 0.0, diff
print("[ok] original encoder == data/checkpoints/encoder.pth")
