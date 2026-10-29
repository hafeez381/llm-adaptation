"""C25: agreement-based and quality-aware stopping, simulated over the saved rollout orders.

python -m task3.adaptive --dataset {math,mathvista} --phase {dev,heldout}
dev:     grid over tau (and gamma), pick per family the cheapest setting within 1 pp of fixed-32 self-consistency,
         write task3/configs/adaptive_thresholds.json (commit it before the heldout phase).
heldout: refuses to run unless that file is committed and unchanged, then evaluates the frozen settings.
Stopping sees revealed candidates only; both controllers return the current self-consistency answer.
"""
import argparse
import json
import pickle
import subprocess
from collections import Counter

import numpy as np

from common.guards import GuardError
from common.results import ROOT, read_json, write_json
from task3.load import equiv_fn
from task3.offline import finite, prefix_fields
from task3.selection import NO_ANSWER, greedy_groups, rwc, rwc_margin, sc_leader
from task3.signals import visconf

CHECKS, MAX_N = [4, 8, 16], 32
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
THRESH = ROOT / "task3/configs/adaptive_thresholds.json"


def quality(f: dict, vlm: bool) -> np.ndarray:
    return finite(visconf(f["kl"], f["v"])[0]) if vlm else finite(f["K"])


def simulate(q, order, gid_full, policy: dict, vlm: bool) -> dict:
    """policy: {"family": "agreement"|"quality"|"always32", "tau", "gamma"}."""
    stop = MAX_N
    if policy["family"] != "always32":
        for n in CHECKS:
            f = prefix_fields(q, order, gid_full, n, vlm)
            pos, support, unique = sc_leader(f["groups"])
            ok = unique and f["keys"][pos] != NO_ANSWER and support >= policy["tau"] * n
            if ok and policy["family"] == "quality":
                s = quality(f, vlm)
                ok = f["groups"][rwc(f["groups"], s)] == f["groups"][pos] and rwc_margin(f["groups"], s) >= policy["gamma"]
            if ok:
                stop = n
                break
    f = prefix_fields(q, order, gid_full, stop, vlm)
    pos, _, _ = sc_leader(f["groups"])
    return {"stop": stop, "correct": bool(q["correct"][order[pos]]), "cost": int(sum(f["T"])),
            "rollout": order[pos]}


def policies() -> list[dict]:
    out = [{"family": "always32", "tau": None, "gamma": None}]
    out += [{"family": "agreement", "tau": t, "gamma": None} for t in TAUS]
    out += [{"family": "quality", "tau": t, "gamma": g} for t in TAUS for g in GAMMAS]
    return out


def evaluate(qs, orders, sids, pols, eq, vlm) -> dict:
    """Per policy: mean over questions of the mean over orders (accuracy, cost, rollouts) plus per-question rows."""
    by_id = {q["sid"]: q for q in qs}
    res = {i: {"acc": [], "cost": [], "n": [], "rows": []} for i in range(len(pols))}
    for sid in sids:
        q = by_id[sid]
        for order in orders[sid]:
            gid = greedy_groups([q["keys"][i] for i in order], eq)
            for i, p in enumerate(pols):
                res[i]["rows"].append({"sid": sid, **simulate(q, order, gid, p, vlm)})
    out = []
    for i, p in enumerate(pols):
        rows = res[i]["rows"]
        per_q = {}
        for r in rows:
            per_q.setdefault(r["sid"], []).append(r)
        acc = np.mean([np.mean([r["correct"] for r in v]) for v in per_q.values()])
        cost = np.mean([np.mean([r["cost"] for r in v]) for v in per_q.values()])
        nro = np.mean([np.mean([r["stop"] for r in v]) for v in per_q.values()])
        out.append(p | {"acc": float(acc), "cost": float(cost), "rollouts": float(nro),
                        "stop_counts": dict(Counter(r["stop"] for r in rows)), "rows": rows})
    return out


def pick(results: list[dict], family: str, sc32: float) -> dict:
    cand = [r for r in results if r["family"] in (family, "always32") and r["acc"] >= sc32 - 0.01]
    key = lambda r: (r["cost"], -r["acc"], r["tau"] if r["tau"] is not None else 9, r["gamma"] or 0)
    best = min(cand, key=key)
    return {k: v for k, v in best.items() if k != "rows"}


def assert_dev_heldout() -> str:
    rel = str(THRESH.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (THRESH.exists() and tracked and clean):
        raise GuardError("thresholds must be committed and unchanged before held-out scoring")
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista"], required=True)
    ap.add_argument("--phase", choices=["dev", "heldout"], required=True)
    args = ap.parse_args()
    ds, vlm = args.dataset, args.dataset == "mathvista"
    with open(ROOT / f"data/cache/task3/{ds}.pkl", "rb") as f:
        qs = pickle.load(f)
    split = read_json(ROOT / f"data/task3_split_{ds}.json")
    orders = read_json(ROOT / f"data/task3_orders_{ds}.json")
    eq = equiv_fn(ds)
    out = ROOT / "results/task3/adaptive"
    if args.phase == "dev":
        res = evaluate(qs, orders, split["dev"], policies(), eq, vlm)
        sc32 = next(r["acc"] for r in res if r["family"] == "always32")
        chosen = {f: pick(res, f, sc32) for f in ("agreement", "quality")}
        th = read_json(THRESH) if THRESH.exists() else {}
        th[ds] = {"dev_sc32_acc": sc32, "rule": "lowest mean token cost with dev acc >= fixed-32 SC - 1pp; "
                  "ties: higher acc, smaller tau, smaller gamma", "selected": chosen}
        write_json(THRESH, th)
        write_json(out / f"{ds}_dev_grid.json", [{k: v for k, v in r.items() if k != "rows"} for r in res])
        print(json.dumps(th[ds], indent=1))
    else:
        sha = assert_dev_heldout()
        frozen = read_json(THRESH)[ds]["selected"]
        pols = [{"family": "always32", "tau": None, "gamma": None}] + \
               [{"family": frozen[f]["family"], "tau": frozen[f]["tau"], "gamma": frozen[f]["gamma"]} for f in ("agreement", "quality")]
        res = evaluate(qs, orders, split["heldout"], pols, eq, vlm)
        names = ["always32", "agreement", "quality"]
        summary = {"thresholds_commit": sha}
        for name, r in zip(names, res):
            summary[name] = {k: v for k, v in r.items() if k != "rows"}
            with open(out / f"{ds}_heldout_{name}_rows.jsonl", "w") as fh:
                fh.writelines(json.dumps(x) + "\n" for x in r["rows"])
        # incorrect early stops, and those where fixed-32 self-consistency was correct
        full = {(x["sid"], i % 10): x for i, x in enumerate(res[0]["rows"])}
        for name, r in zip(names[1:], res[1:]):
            early = [(i, x) for i, x in enumerate(r["rows"]) if x["stop"] < MAX_N]
            wrong = [(i, x) for i, x in early if not x["correct"]]
            summary[name]["early_stops"] = len(early)
            summary[name]["incorrect_early_stops"] = len(wrong)
            summary[name]["incorrect_early_where_sc32_correct"] = sum(full[(x["sid"], i % 10)]["correct"] for i, x in wrong)
        write_json(out / f"{ds}_heldout_summary.json", summary)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
