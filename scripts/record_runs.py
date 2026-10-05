"""
Training server: write results/runs/<task>/<train_config>.json for every finished ACT_TacEnc training
(<UniVTAC>/data/act_tacenc/act_ckpt/act-<task>/<data_version>-<N>/<train_config>/policy_last.ckpt).

The record pins what was trained (checkpoint sha256, train config, raw episodes, repo commit, train log) so the
evaluation server can verify it got the same checkpoint (scripts/ckpt_transfer.py import, scripts/eval_run.py).

    python scripts/record_runs.py            # all finished runs (existing records with the same sha256 are kept)
"""
import hashlib
import json
import os
import re
import socket
from pathlib import Path

import yaml

JEPA_ROOT = Path(__file__).resolve().parents[1]
UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))
OUT_ROOT = Path(os.environ.get("ACT_TACENC_OUT", str(UNIVTAC_ROOT / "data" / "act_tacenc")))
RESULTS = Path(os.environ.get("RESULTS_DIR", str(JEPA_ROOT / "results")))
FILES = ["policy_last.ckpt", "dataset_stats.pkl"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_log_info(task, cfg):
    """Newest training log of this run that ended with exit=0 (written by train_queue.sh)."""
    logs = sorted((OUT_ROOT / "logs").glob(f"{task}_{cfg}_seed*_*.log"), reverse=True)
    for log in logs:
        text = log.read_text(errors="ignore").replace("\r", "\n")
        if not re.search(r"^exit=0 ", text, re.M):
            continue
        head = dict(kv.split("=", 1) for kv in text.splitlines()[0].split() if "=" in kv)
        end = re.search(r"^exit=0 end=(.+)$", text, re.M).group(1)
        progress = re.findall(r"(\d+)/(\d+) \[[^\]]*, epoch=(\d+), loss=([\d.]+)\]", text)
        val = re.search(r"Best ckpt, val loss ([\d.]+) @ epoch(\d+)", text)
        return {
            "log": log.name, "repo": head.get("repo"), "gpu": head.get("gpu"), "seed": head.get("seed"),
            "start": " ".join(text.splitlines()[0].split("start=")[1:]) or None, "end": end,
            "last_progress": {"step": int(progress[-1][0]), "of": int(progress[-1][1]), "epoch": int(progress[-1][2]),
                              "train_loss": float(progress[-1][3])} if progress else None,
            "best_val": {"loss": float(val.group(1)), "epoch": int(val.group(2))} if val else None,
        }
    return None


def main():
    sim_cfgs_path = OUT_ROOT / "SIM_TASK_CONFIGS.json"
    sim_cfgs = json.loads(sim_cfgs_path.read_text()) if sim_cfgs_path.exists() else {}
    n_new = n_same = 0
    for last in sorted((OUT_ROOT / "act_ckpt").glob("act-*/*-*/train_config_*/policy_last.ckpt")):
        ckpt = last.parent
        cfg, data, task = ckpt.name, ckpt.parent.name, ckpt.parent.parent.name[len("act-"):]
        version, ep = data.rsplit("-", 1)
        files = {f: sha256(ckpt / f) for f in FILES if (ckpt / f).exists()}
        out = RESULTS / "runs" / task / f"{cfg}.json"
        if out.exists() and json.loads(out.read_text())["files"].get("policy_last.ckpt") == files["policy_last.ckpt"]:
            n_same += 1
            continue
        yml = UNIVTAC_ROOT / "policy" / "ACT_TacEnc" / f"{cfg}.yml"
        train_cfg = yaml.safe_load(yml.read_text())
        key = f"sim-{task}-{version}-{ep}"
        record = {
            "task": task, "train_config": cfg, "data_version": version, "ep_num": int(ep),
            "files": files, "checkpoint_bytes": last.stat().st_size,
            "tactile_encoder": train_cfg.get("tactile_encoder"),
            "freeze_tactile_encoder": train_cfg.get("freeze_tactile_encoder"),
            "num_steps": train_cfg.get("num_steps"),
            "num_steps_note": "imitate_episodes.py runs num_steps + 1 optimizer steps (same as policy/ACT)",
            "train_config_sha256": sha256(yml), "train_config_content": train_cfg,
            "raw_episodes": sim_cfgs.get(key, {}).get("raw_episodes"),
            "train_log": train_log_info(task, cfg),
            "host": socket.gethostname(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print(f"!! {out}: 체크포인트가 바뀌었습니다 (재학습) — 기록을 덮어씁니다")
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        n_new += 1
        print(f"[record] {task} {cfg}: sha256 {files['policy_last.ckpt'][:12]}…  log {(record['train_log'] or {}).get('log')}")
    print(f"[record] new/updated {n_new}, unchanged {n_same} → {RESULTS / 'runs'}")


if __name__ == "__main__":
    main()
