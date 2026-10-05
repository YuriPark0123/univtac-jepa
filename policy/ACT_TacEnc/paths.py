"""
Where ACT_TacEnc reads UniVTAC files and writes its outputs.

The code lives in the univtac-jepa repo and is linked into <UniVTAC>/policy/ACT_TacEnc, so paths are not derived
from __file__. Heavy outputs (processed episodes, checkpoints) go under <UniVTAC>/data/act_tacenc on the local disk,
never into the code repo.
    UNIVTAC_ROOT    (default /workspace/UniVTAC, the container path used on every server)
    ACT_TACENC_OUT  (default <UNIVTAC_ROOT>/data/act_tacenc)
"""
import os
from pathlib import Path

UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))
UNIVTAC_POLICY = UNIVTAC_ROOT / "policy"
OUT_ROOT = Path(os.environ.get("ACT_TACENC_OUT", str(UNIVTAC_ROOT / "data" / "act_tacenc")))
SIM_TASK_CONFIGS_PATH = OUT_ROOT / "SIM_TASK_CONFIGS.json"
CKPT_ROOT = OUT_ROOT / "act_ckpt"


def ckpt_dir(task_name, data_version, ep_num, train_config):
    return CKPT_ROOT / f"act-{task_name}" / f"{data_version}-{ep_num}" / train_config
