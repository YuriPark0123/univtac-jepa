"""
Evaluate one (task, method) on the fixed seed list of results/protocol.json and write per-seed results.

Same rollout loop as UniVTAC scripts/eval_policy.py, with three differences that keep seed sets comparable:
  * the seed list is fixed (protocol.json), it does not drift when an episode errors
  * an errored episode (e.g. "reset exceed time limit" when the GPU is shared) is retried on the SAME seed
    (protocol max_retry); the number of attempts is recorded
  * results are appended per seed, so an interrupted run resumes where it stopped

    python scripts/eval_run.py <task> <method> [--shard i/n] [--headless]
methods:
    public_univtac, public_vision_only   released UniVTAC ACT checkpoints (data/checkpoints/<task>/{univtac,vision_only})
    <name> (e.g. original, sparsh_ijepa)  ACT_TacEnc checkpoint trained with train_config_<name>.yml
outputs:
    results/evals/<task>/<method>/per_seed.shard<i>of<n>.csv   (committed)
    results/evals/<task>/<method>/eval.shard<i>of<n>.json      (committed: checkpoint sha256, versions, host, timing)
    <UniVTAC>/data/act_tacenc/eval_raw/<task>/<method>/...      (local: videos, simulator logs)
"""
import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[1]
UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))
OUT_ROOT = Path(os.environ.get("ACT_TACENC_OUT", str(UNIVTAC_ROOT / "data" / "act_tacenc")))
RESULTS = Path(os.environ.get("RESULTS_DIR", str(JEPA_ROOT / "results")))
PUBLIC = {"public_univtac": ("univtac", "train_config"), "public_vision_only": ("vision_only", "train_config_vision")}
FIELDS = ["seed", "result", "success", "steps", "actions", "cost_time", "attempts", "error"]

parser = argparse.ArgumentParser(description="UniVTAC fixed-seed evaluation")
parser.add_argument("task_name")
parser.add_argument("method")
parser.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th seed starting at i")
parser.add_argument("--protocol", default=str(RESULTS / "protocol.json"))
parser.add_argument("--limit", type=int, default=0, help="only the first N seeds of this shard (smoke tests)")
parser.add_argument("--tag", default="", help="suffix of the output folder, e.g. '_smoke'")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib  # noqa: E402

import yaml  # noqa: E402

os.chdir(UNIVTAC_ROOT)
sys.path[:0] = [str(UNIVTAC_ROOT), str(UNIVTAC_ROOT / "policy")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path):
    try:
        rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(path), "diff", "--quiet", "HEAD"]) != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve_method(task, method, proto):
    """-> policy module name, deploy.yml path, env vars for the deploy code, checkpoint dir"""
    ep, ver = str(proto["ep_num"]), proto["data_version"]
    if method in PUBLIC:
        folder, train_config = PUBLIC[method]
        src = UNIVTAC_ROOT / "data" / "checkpoints" / task / folder
        if not (src / "policy_last.ckpt").exists():
            sys.exit(f"!! {src}/policy_last.ckpt 없음 — 먼저: bash data/download.sh --checkpoint {task}")
        # policy/ACT/deploy_policy.py reads policy/ACT/act_ckpt/act-<task>/<task_config>-<EP_NUM>/<TRAIN_CONFIG>
        link = UNIVTAC_ROOT / "policy" / "ACT" / "act_ckpt" / f"act-{task}" / f"{proto['task_config']}-{ep}" / train_config
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(src)
        assert link.resolve() == src.resolve(), f"{link} -> {link.resolve()}, expected {src}"
        return "ACT", UNIVTAC_ROOT / "policy" / "ACT" / "deploy.yml", {"TRAIN_CONFIG": train_config, "EP_NUM": ep}, src
    train_config = f"train_config_{method}"
    ckpt = OUT_ROOT / "act_ckpt" / f"act-{task}" / f"{ver}-{ep}" / train_config
    if not (ckpt / "policy_last.ckpt").exists():
        sys.exit(f"!! {ckpt}/policy_last.ckpt 없음 — 학습하거나 scripts/ckpt_transfer.py import 로 가져오세요")
    env = {"TRAIN_CONFIG": train_config, "EP_NUM": ep, "DATA_VERSION": ver}
    return "ACT_TacEnc", UNIVTAC_ROOT / "policy" / "ACT_TacEnc" / "deploy.yml", env, ckpt


def main():
    proto = json.loads(Path(args_cli.protocol).read_text())
    task_name, method = args_cli.task_name, args_cli.method
    shard_i, shard_n = (int(x) for x in args_cli.shard.split("/"))
    seeds = proto["seeds"][shard_i::shard_n]
    if args_cli.limit:
        seeds = seeds[:args_cli.limit]

    policy_name, deploy_file, env, ckpt = resolve_method(task_name, method, proto)
    os.environ.update(env)

    # expected checkpoint (results/runs/<task>/<train_config>.json written on the training server)
    ckpt_sha = sha256(ckpt / "policy_last.ckpt")
    run_record = RESULTS / "runs" / task_name / f"{env['TRAIN_CONFIG']}.json"
    if policy_name == "ACT_TacEnc":
        if not run_record.exists():
            print(f"!! 경고: {run_record} 없음 — 체크포인트가 어떤 학습에서 왔는지 검증할 수 없습니다")
        elif json.loads(run_record.read_text())["files"]["policy_last.ckpt"] != ckpt_sha:
            sys.exit(f"!! {ckpt}/policy_last.ckpt sha256 이 {run_record} 와 다릅니다 (다른 학습의 체크포인트)")

    out_dir = RESULTS / "evals" / task_name / f"{method}{args_cli.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard{shard_i}of{shard_n}"
    csv_path, meta_path = out_dir / f"per_seed.{stem}.csv", out_dir / f"eval.{stem}.json"
    other = [p.name for p in out_dir.glob("per_seed.shard*of*.csv") if not p.name.endswith(f"of{shard_n}.csv")]
    if other:
        sys.exit(f"!! {out_dir} 에 다른 shard 수로 만든 결과가 있습니다 ({other}) — 같은 shard 수로 이어서 하거나 폴더를 정리하세요")

    done = {}
    if csv_path.exists():
        with open(csv_path) as f:
            done = {int(r["seed"]): r for r in csv.DictReader(f)}
    todo = [s for s in seeds if s not in done]
    print(f"[eval_run] {task_name} / {method}: shard {shard_i}/{shard_n}, seeds {len(seeds)}, done {len(done)}, todo {len(todo)}")

    deploy_config = yaml.safe_load(deploy_file.read_text())
    task_config = yaml.safe_load((UNIVTAC_ROOT / "task_config" / f"{proto['task_config']}.yml").read_text())
    deploy_config.update(task_name=task_name, task_config=proto["task_config"], train_config_name=env["TRAIN_CONFIG"],
                         expert_data_num=env["EP_NUM"], instuction_file=None,
                         eval_steps_per_data_frame=proto["eval_steps_per_data_frame"])
    if "DATA_VERSION" in env:
        deploy_config["data_version"] = env["DATA_VERSION"]
    instructions = {"seen": ["Empty"], "unseen": ["Empty"]}  # same as the released logs (instuction_file: null)

    task_module = importlib.import_module(f"envs.{task_name}")
    policy_module = importlib.import_module(f"policy.{policy_name}")
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = OUT_ROOT / "eval_raw" / task_name / f"{method}{args_cli.tag}" / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if proto.get("reset_time_limit") is not None:
        env_cfg.reset_time_limit = float(proto["reset_time_limit"])

    policy = policy_module.Policy(deploy_config)
    task = task_module.Task(env_cfg, mode="eval")

    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                                      text=True).strip()
    except Exception:
        gpu = "unknown"
    from importlib.metadata import version
    meta = {
        "task": task_name, "method": method, "shard": args_cli.shard, "seeds": len(seeds),
        "policy": policy_name, "train_config": env["TRAIN_CONFIG"], "checkpoint_dir": str(ckpt),
        "checkpoint_sha256": ckpt_sha, "run_record": str(run_record.relative_to(JEPA_ROOT)) if run_record.exists() else None,
        "protocol_sha256": sha256(args_cli.protocol), "repo": git_rev(JEPA_ROOT), "univtac": git_rev(UNIVTAC_ROOT),
        "isaacsim": version("isaacsim"), "isaaclab": version("isaaclab"), "host": socket.gethostname(),
        "gpu_at_start": gpu, "reset_time_limit": env_cfg.reset_time_limit, "max_retry": proto["max_retry"],
        "raw_dir": str(env_cfg.save_dir), "start": time.strftime("%F %T"), "end": None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for n_done, seed in enumerate(todo, 1):
            row, errors = None, []
            for attempt in range(1, proto["max_retry"] + 2):
                t0 = time.perf_counter()
                succ = False
                try:
                    task.mode = "eval"
                    task.reset(seed=seed, instructions=instructions[proto["instruction_type"]])
                    task.mean_steps = task.cfg.step_lim
                    policy.reset()
                    while task.take_action_cnt < task.cfg.step_lim:
                        observation = task._get_observations()
                        policy.eval(task, observation)
                        if task.eval_success:
                            succ = True
                            break
                        if task.check_early_stop():
                            break
                except Exception as e:  # retried on the same seed
                    errors.append(f"attempt {attempt}: {type(e).__name__}: {e}"[:300])
                    print(f"[eval_run] seed {seed} attempt {attempt} error:\n{traceback.format_exc()}")
                    task.clean_cache(result="error")
                    continue
                result = "success" if succ else "failed"
                task.clean_cache(result=result)
                row = dict(seed=seed, result=result, success=int(succ), steps=task.step_count,
                           actions=task.take_action_cnt, cost_time=round(time.perf_counter() - t0, 2),
                           attempts=attempt, error=" | ".join(errors))
                break
            if row is None:
                row = dict(seed=seed, result="error", success="", steps="", actions="", cost_time="",
                           attempts=proto["max_retry"] + 1, error=" | ".join(errors))
            writer.writerow(row)
            f.flush()
            print(f"[eval_run] [{n_done}/{len(todo)}] seed {seed} {row['result']} (attempts {row['attempts']}, {row['cost_time']} s)")

    meta["end"] = time.strftime("%F %T")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    task.close()
    policy.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
