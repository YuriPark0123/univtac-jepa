"""
Reload the checkpoint trained by run_tests.sh through the evaluation entry point (policy.ACT_TacEnc.Policy) and check
  * strict state-dict match, * which recorded tactile frames are fed at each eval step, * action shape.
Expects TRAIN_CONFIG / EP_NUM / DATA_VERSION / ACT_TACENC_OUT in the environment (set by run_tests.sh).
"""
import os
import sys

import torch

ROOT = os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC")
sys.path[:0] = [ROOT, f"{ROOT}/policy"]
os.chdir(ROOT)
from policy.ACT_TacEnc.deploy_policy import Policy  # noqa: E402
from policy.ACT_TacEnc.paths import ckpt_dir  # noqa: E402

args = {"task_name": "grasp_classify", "task_config": "demo", "seed": 0,
        "data_version": os.environ["DATA_VERSION"], "eval_steps_per_data_frame": 2}
p = Policy(args)
path = ckpt_dir("grasp_classify", os.environ["DATA_VERSION"], os.environ["EP_NUM"], os.environ["TRAIN_CONFIG"])
p.model.policy.load_state_dict(torch.load(path / "policy_last.ckpt"), strict=True)
print(f"[ok] strict reload from {path}")

# feed frames whose pixel value = eval step, read back which steps the encoder input is built from
p.reset()
picked = []
for step in range(14):
    tac = torch.full((240, 320, 3), step, dtype=torch.uint8, device="cuda")
    obs = {"observation": {"head": {"rgb": torch.zeros(270, 480, 3, device="cuda")}},
           "tactile": {"left_tactile": {"rgb_marker": tac}, "right_tactile": {"rgb_marker": tac}},
           "embodiment": {"joint": torch.zeros(9, device="cuda")}}
    enc = p.encode_obs(obs)
    picked.append([round(v * 255) for v in enc["tac_left"][:, 0, 0, 0].tolist()])
    action = p.model.get_action(enc)

# sparsh_ijepa: offsets (0, -5 data frames, first); 1 data frame = 2 eval steps
assert p.frame_offsets == (0, -5, "first"), p.frame_offsets
assert picked[13] == [13, 3, 0] and picked[11] == [11, 1, 0] and picked[5] == [5, 0, 0], picked
assert action.shape == (1, 8), action.shape
print(f"[ok] tactile frames at eval step 13 -> {picked[13]} (t, t-5 data frames, background); action {action.shape}")
