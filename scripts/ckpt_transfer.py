"""
Move trained ACT_TacEnc checkpoints between servers through a shared folder (e.g. NAS), verified by sha256.

    # training server (after scripts/record_runs.py and git push of results/runs)
    python scripts/ckpt_transfer.py export /mnt/nas/yuri/univtac_ckpt [--task insert_hole]
    # evaluation server (after git pull)
    python scripts/ckpt_transfer.py import /mnt/nas/yuri/univtac_ckpt [--task insert_hole]

Only runs listed in results/runs/<task>/<train_config>.json are moved, and only policy_last.ckpt + dataset_stats.pkl
(~383 MB per run). Files are copied by content only (no permissions / symlinks), which works on CIFS.
Inside the container the shared folder must be mounted; otherwise run this on the host with UNIVTAC_ROOT=<host path>.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[1]
UNIVTAC_ROOT = Path(os.environ.get("UNIVTAC_ROOT", "/workspace/UniVTAC"))
OUT_ROOT = Path(os.environ.get("ACT_TACENC_OUT", str(UNIVTAC_ROOT / "data" / "act_tacenc")))
RESULTS = Path(os.environ.get("RESULTS_DIR", str(JEPA_ROOT / "results")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_dir(rec):
    return Path("act_ckpt") / f"act-{rec['task']}" / f"{rec['data_version']}-{rec['ep_num']}" / rec["train_config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    ap.add_argument("shared_dir", type=Path)
    ap.add_argument("--task", action="append", help="limit to these tasks (repeatable)")
    args = ap.parse_args()

    records = [json.loads(p.read_text()) for p in sorted((RESULTS / "runs").glob("*/*.json"))]
    records = [r for r in records if not args.task or r["task"] in args.task]
    if not records:
        raise SystemExit(f"!! {RESULTS / 'runs'} 에 기록이 없습니다 (학습 서버: scripts/record_runs.py → git push / 평가 서버: git pull)")

    ok = bad = skipped = 0
    for rec in records:
        src_root, dst_root = (OUT_ROOT, args.shared_dir) if args.mode == "export" else (args.shared_dir, OUT_ROOT)
        name = f"{rec['task']} {rec['train_config']}"
        for fname, expected in rec["files"].items():
            src, dst = src_root / rel_dir(rec) / fname, dst_root / rel_dir(rec) / fname
            if dst.exists() and sha256(dst) == expected:
                skipped += 1
                continue
            if not src.exists():
                print(f"!! 없음: {src}")
                bad += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            if sha256(tmp) != expected:
                tmp.unlink()
                print(f"!! sha256 불일치 (원본이 기록과 다름): {src}")
                bad += 1
                continue
            os.replace(tmp, dst)
            ok += 1
            print(f"[{args.mode}] {name}: {fname} ✓ sha256")
    print(f"[{args.mode}] 복사 {ok}, 이미 같음 {skipped}, 실패 {bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
