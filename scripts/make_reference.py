"""
Convert the evaluation logs released with the public UniVTAC checkpoints into results/reference/<task>/<variant>/
per_seed.csv (same columns as scripts/eval_run.py), so re-evaluations can be compared seed by seed.

    python scripts/make_reference.py            # tasks of results/protocol.json; downloads only log.log (~15 KB each)
"""
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[1]
UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))
RESULTS = Path(os.environ.get("RESULTS_DIR", str(JEPA_ROOT / "results")))
MODELSCOPE = os.environ.get("MODELSCOPE_BIN", "/root/miniconda3/envs/UniVTAC/bin/modelscope")
FIELDS = ["seed", "result", "success", "steps", "actions", "cost_time", "attempts", "error"]
EPISODE = re.compile(r"\] \[\s*\d+\s*\] Seed (\d+) (success|failed) after ([\d.]+) s\.\s*\nsteps:\s*(\d+)\s*, actions:\s*(\d+)")
ERROR = re.compile(r"\] \[\s*\d+\s*\] Seed (\d+) occurred exception: (.*)")


def parse(log_text):
    rows = {}
    for seed, err in ERROR.findall(log_text):
        rows[int(seed)] = dict(seed=int(seed), result="error", success="", steps="", actions="", cost_time="",
                               attempts=1, error=err.strip()[:300])
    for seed, res, t, steps, actions in EPISODE.findall(log_text):
        rows[int(seed)] = dict(seed=int(seed), result=res, success=int(res == "success"), steps=int(steps),
                               actions=int(actions), cost_time=float(t), attempts=1, error="")
    cfg = json.loads(re.search(r"Eval Config: (\{.*?\n\})", log_text, re.S).group(1))
    final = re.search(r"Final Result: (.*)", log_text)
    return [rows[s] for s in sorted(rows)], cfg, final.group(1).strip() if final else None


def main():
    proto = json.loads((RESULTS / "protocol.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for task in proto["tasks"]:
            local = UNIVTAC_ROOT / "data" / "checkpoints" / task
            if not all((local / v / "log.log").exists() for v in ("univtac", "vision_only")):
                subprocess.run([MODELSCOPE, "download", "byml2024/UniVTAC", "--repo-type", "dataset",
                                "--include", f"checkpoints/{task}/*/log.log", "--local-dir", tmp],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                local = Path(tmp) / "checkpoints" / task
            for variant in ("univtac", "vision_only"):
                rows, cfg, final = parse((local / variant / "log.log").read_text())
                out = RESULTS / "reference" / task / f"public_{variant}"
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "per_seed.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                (out / "source.json").write_text(json.dumps({
                    "source": f"modelscope byml2024/UniVTAC checkpoints/{task}/{variant}/log.log",
                    "eval_config": cfg, "final_result": final, "episodes": len(rows)}, indent=2) + "\n")
                print(f"[reference] {task:20s} public_{variant:12s} {final}  ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
