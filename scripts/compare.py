"""
Build the comparison tables from per-seed results only (no hand-copied numbers).

    python scripts/compare.py        # -> results/tables/summary.md, summary.csv

Inputs:  results/protocol.json, results/evals/<task>/<method>/per_seed.*.csv + eval.*.json,
         results/reference/<task>/public_*/per_seed.csv, results/runs/<task>/<train_config>.json
Validity (an eval enters the main table only if all hold):
         same protocol (sha256), seed set == protocol seeds, no 'error' seed after retries,
         one checkpoint sha256 across shards, and for ACT_TacEnc runs: sha256 == results/runs record
Stats:   success k/n, Wilson 95% CI, delta vs baseline (protocol baseline_method) with exact McNemar p on the
         paired seeds, mean steps of successful episodes; for public_* re-evals: per-seed agreement with the
         released log.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("RESULTS_DIR", str(JEPA_ROOT / "results")))
ORDER = ["public_vision_only", "public_univtac", "vision", "original", "original_frozen", "resnet18_imagenet",
         "sparsh_mae", "sparsh_dino", "sparsh_ijepa", "sparsh_vjepa"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """two-sided exact McNemar = binomial test on the discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def read_rows(paths):
    rows = {}
    dup = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                s = int(r["seed"])
                if s in rows:
                    dup.append(s)
                rows[s] = r
    return rows, dup


def load_evals(proto, proto_sha):
    evals = {}
    for d in sorted((RESULTS / "evals").glob("*/*")):
        task, method = d.parent.name, d.name
        rows, dup = read_rows(sorted(d.glob("per_seed.*.csv")))
        metas = [json.loads(p.read_text()) for p in sorted(d.glob("eval.*.json"))]
        problems = []
        if dup:
            problems.append(f"seed 중복 {len(dup)}개")
        missing = set(proto["seeds"]) - set(rows)
        extra = set(rows) - set(proto["seeds"])
        if missing:
            problems.append(f"seed 누락 {len(missing)}개")
        if extra:
            problems.append(f"protocol 밖 seed {len(extra)}개")
        n_err = sum(r["result"] == "error" for r in rows.values())
        if n_err:
            problems.append(f"error {n_err}개")
        if {m["protocol_sha256"] for m in metas} != {proto_sha}:
            problems.append("protocol 불일치")
        shas = {m["checkpoint_sha256"] for m in metas}
        if len(shas) != 1:
            problems.append("shard 간 체크포인트 다름")
        rec_path = RESULTS / "runs" / task / f"train_config_{method}.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
        if metas and metas[0]["policy"] == "ACT_TacEnc":
            if rec is None:
                problems.append("학습 기록(results/runs) 없음")
            elif rec["files"]["policy_last.ckpt"] not in shas or len(shas) != 1:
                problems.append("체크포인트가 학습 기록과 다름")
        evals[(task, method)] = dict(rows=rows, metas=metas, problems=problems, run=rec,
                                     retried=sum(int(r["attempts"] or 1) > 1 for r in rows.values()))
    return evals


def load_reference():
    ref = {}
    for d in sorted((RESULTS / "reference").glob("*/*")):
        rows, _ = read_rows([d / "per_seed.csv"])
        ref[(d.parent.name, d.name)] = rows
    return ref


def describe(method, ev):
    """data version, train steps, frozen encoder for the table"""
    if method.startswith("public_"):
        return "isaac45 (README; 확인 못 함)", "공개", "아니오" if method == "public_univtac" else "—"
    rec = ev.get("run") or {}
    frozen = rec.get("freeze_tactile_encoder")
    if method == "vision":
        frozen_s = "—"
    else:
        frozen_s = "예" if frozen else ("아니오" if frozen is False else "?")
    steps = rec.get("num_steps")
    return rec.get("data_version", "?"), (f"{steps}" if steps else "?"), frozen_s


def main():
    proto_path = RESULTS / "protocol.json"
    proto = json.loads(proto_path.read_text())
    proto_sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    seeds = proto["seeds"]
    evals, ref = load_evals(proto, proto_sha), load_reference()
    base_name = proto["baseline_method"]

    md = ["# UniVTAC 촉각 encoder 비교", "",
          f"조건: `results/protocol.json` (seed {seeds[0]}–{seeds[-1]}, n={len(seeds)}, task_config `{proto['task_config']}`, "
          f"data `{proto['data_version']}`, 데모 {proto['ep_num']}개). 기준선(Δ, McNemar): `{base_name}` (같은 파이프라인 재학습). "
          "셀 = 성공/n [Wilson 95% CI]. p = exact McNemar (같은 seed 짝 비교).", ""]
    rows_csv = []
    for task in proto["tasks"]:
        methods = sorted({m for (t, m) in evals if t == task},
                         key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
        md += [f"## {task}", "",
               "| 방법 | 성공 [95% CI] | Δ vs 기준선 | McNemar p | 성공 시 평균 step | 재시도 seed | 공개 로그와 seed 일치 | 데이터 | 학습 step | encoder 고정 | 유효 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        base = evals.get((task, base_name))
        base_ok = base is not None and not base["problems"]
        for (t, name), r in sorted(ref.items()):
            if t != task:
                continue
            k = sum(int(x["success"] or 0) for x in r.values())
            lo, hi = wilson(k, len(r))
            md.append(f"| {name} (공개 log, 참고) | {k}/{len(r)} [{lo*100:.0f}–{hi*100:.0f}] | | | | | | isaac45 (README) | 공개 | | 참고 |")
        for m in methods:
            ev = evals[(task, m)]
            rows = ev["rows"]
            valid = not ev["problems"]
            n = sum(1 for s in seeds if s in rows and rows[s]["result"] != "error")
            k = sum(1 for s in seeds if s in rows and rows[s]["result"] == "success")
            lo, hi = wilson(k, n)
            delta = p = ""
            if base_ok and valid and m != base_name:
                br = base["rows"]
                b = sum(1 for s in seeds if rows[s]["result"] == "success" and br[s]["result"] != "success")
                c = sum(1 for s in seeds if rows[s]["result"] != "success" and br[s]["result"] == "success")
                kb = sum(1 for s in seeds if br[s]["result"] == "success")
                delta, p = f"{k - kb:+d}", f"{mcnemar_exact(b, c):.3f}"
            succ_steps = [int(rows[s]["actions"]) for s in seeds if s in rows and rows[s]["result"] == "success"]
            mean_steps = f"{sum(succ_steps) / len(succ_steps):.1f}" if succ_steps else ""
            agree = ""
            if (task, m) in ref:
                rr = ref[(task, m)]
                common = [s for s in seeds if s in rows and s in rr and rows[s]["result"] != "error"]
                if common:
                    same = sum((rows[s]["result"] == "success") == (rr[s]["result"] == "success") for s in common)
                    agree = f"{same}/{len(common)}"
            data, tsteps, frozen = describe(m, ev)
            mark = "✅" if valid else "❌ " + ", ".join(ev["problems"])
            md.append(f"| {m} | {k}/{n} [{lo*100:.0f}–{hi*100:.0f}] | {delta} | {p} | {mean_steps} | {ev['retried']} | {agree} | "
                      f"{data} | {tsteps} | {frozen} | {mark} |")
            rows_csv.append(dict(task=task, method=m, success=k, n=n, ci_low=round(lo, 4), ci_high=round(hi, 4),
                                 delta_vs_baseline=delta, mcnemar_p=p, mean_actions_success=mean_steps,
                                 retried_seeds=ev["retried"], agreement_with_public_log=agree, data_version=data,
                                 train_steps=tsteps, encoder_frozen=frozen, valid=valid, problems="; ".join(ev["problems"]),
                                 checkpoint_sha256=",".join(sorted({x["checkpoint_sha256"] for x in ev["metas"]})),
                                 eval_hosts=",".join(sorted({x["host"] for x in ev["metas"]}))))
        if not base_ok:
            md.append(f"\n기준선 `{base_name}` 평가가 없거나 무효라 Δ/p 를 계산하지 않았습니다.")
        md.append("")
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    if rows_csv:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    print("\n".join(md))


if __name__ == "__main__":
    main()
