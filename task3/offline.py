"""C21-C23: repeated sampling, selection rules and signal analysis on the supplied pools.

python -m task3.offline --dataset {math,mathvista}
Per question, order and budget N, every rule sees only the first N rollouts of that order (keys, token counts,
scores, calibration). Gold enters only through `correct`, which is read after a rule has chosen.
Writes results/task3/selection/<ds>_rows.jsonl (per question x N x method, averaged over the ten orders),
<ds>_summary.json (recomputed from the rows), <ds>_bootstrap.json and results/task3/signals/<ds>.json.
"""
import argparse
import json
import pickle

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common.results import ROOT, read_json, read_jsonl, write_json
from common.seeding import SEED
from task3.load import equiv_fn
from task3.selection import greedy_groups, highest, rwc, self_consistency
from task3.signals import calibrated_conf, visconf

BUDGETS = [1, 2, 4, 8, 16, 32]
BASE_METHODS = ["first", "sc", "logp_high", "sc_high", "sc_rwc", "shortest"]   # sc_* = self-certainty
VLM_METHODS = ["vis_high", "vis_rwc", "visconf_high", "visconf_rwc", "tokfuse_high", "tokfuse_rwc"]


def finite(x) -> np.ndarray:
    """Rollouts with no tokens have undefined scores; they rank last."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, -np.inf)


def assert_no_gold_in_selection(fields: dict) -> None:
    from common.guards import GuardError
    if {"correct", "gold", "gold_key"} & set(fields):
        raise GuardError(f"selection received gold fields: {set(fields)}")


def select_all(fields: dict, vlm: bool) -> dict[str, int]:
    """Position (within the revealed prefix) chosen by every rule. `fields` holds candidate data only."""
    assert_no_gold_in_selection(fields)
    g, T = fields["groups"], np.asarray(fields["T"])
    L, K = finite(fields["L"]), finite(fields["K"])
    out = {"first": 0, "sc": self_consistency(g), "logp_high": highest(L), "sc_high": highest(K),
           "sc_rwc": rwc(g, K), "shortest": int(np.flatnonzero(T == T.min())[0])}
    if vlm:
        V = finite(fields["V"])
        z, zt = visconf(fields["kl"], fields["v"])
        z, zt = finite(z), finite(zt)
        out |= {"vis_high": highest(V), "vis_rwc": rwc(g, V), "visconf_high": highest(z), "visconf_rwc": rwc(g, z),
                "tokfuse_high": highest(zt), "tokfuse_rwc": rwc(g, zt)}
    return out


def prefix_fields(q: dict, order: list[int], gid_full: list[int], n: int, vlm: bool) -> dict:
    idx = order[:n]
    f = {"groups": gid_full[:n], "T": [q["T"][i] for i in idx], "L": [q["L"][i] for i in idx],
         "K": [q["K"][i] for i in idx], "keys": [q["keys"][i] for i in idx]}
    if vlm:
        f |= {"V": [q["V"][i] for i in idx], "kl": [q["kl"][i] for i in idx], "v": [q["v"][i] for i in idx]}
    return f


def question_rows(q: dict, orders: list[list[int]], split: str, eq, vlm: bool) -> list[dict]:
    methods = BASE_METHODS + (VLM_METHODS if vlm else [])
    acc = {(n, m): [] for n in BUDGETS for m in methods}
    sel = {(n, m): [] for n in BUDGETS for m in methods}
    cost = {(n, m): [] for n in BUDGETS for m in methods}
    oracle, ngroups = {n: [] for n in BUDGETS}, {n: [] for n in BUDGETS}
    for order in orders:
        gid_full = greedy_groups([q["keys"][i] for i in order], eq)
        for n in BUDGETS:
            f = prefix_fields(q, order, gid_full, n, vlm)
            picks = select_all(f, vlm)
            idx = order[:n]
            oracle[n].append(any(q["correct"][i] for i in idx))
            ngroups[n].append(len(set(gid_full[:n])))
            for m, pos in picks.items():
                rid = idx[pos]
                acc[(n, m)].append(q["correct"][rid])
                sel[(n, m)].append(rid)
                cost[(n, m)].append(q["T"][idx[0]] if m == "first" else sum(f["T"]))
    rows = []
    for n in BUDGETS:
        for m in methods:
            a = np.mean(acc[(n, m)])
            o = np.mean(oracle[n])
            rows.append({"sid": q["sid"], "split": split, "N": n, "method": m, "acc": float(a),
                         "oracle": float(o), "n_groups": float(np.mean(ngroups[n])),
                         "cost": float(np.mean(cost[(n, m)])),
                         "fail_sampling": float(1 - o),                    # no correct candidate revealed
                         "fail_selection": float(o - a) if o >= a else 0.0,  # one revealed, rule missed it
                         "selected": sel[(n, m)]})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for split in ("dev", "heldout"):
        for r in rows:
            if r["split"] != split:
                continue
            k = f"{split}|{r['N']}|{r['method']}"
            out.setdefault(k, {"acc": [], "oracle": [], "n_groups": [], "cost": [], "fail_sampling": [], "fail_selection": []})
            for f in out[k]:
                out[k][f].append(r[f])
    return {k: {f: float(np.mean(v)) for f, v in d.items()} | {"questions": len(d["acc"])} for k, d in out.items()}


def bootstrap(rows: list[dict], pairs: list[tuple[str, str]], n_boot=1000) -> dict:
    """95% intervals on held-out accuracy differences, resampling questions with one shared index draw."""
    rng = np.random.default_rng(SEED)
    held = [r for r in rows if r["split"] == "heldout"]
    sids = sorted({r["sid"] for r in held})
    table = {(r["sid"], r["N"], r["method"]): r["acc"] for r in held}
    draws = rng.integers(0, len(sids), size=(n_boot, len(sids)))
    out = {}
    for n in BUDGETS:
        for a, b in pairs:
            if (sids[0], n, a) not in table or (sids[0], n, b) not in table:
                continue
            d = np.array([table[(s, n, a)] - table[(s, n, b)] for s in sids])
            boots = d[draws].mean(1)
            out[f"{n}|{a}-{b}"] = {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
                                   "hi": float(np.percentile(boots, 97.5))}
    return out


def quartile_gap(score, correct) -> float:
    """Top-8 minus bottom-8 correctness rate at N = 32; ties broken by rollout order (stable sort)."""
    o = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    c = np.asarray(correct, dtype=float)
    return float(c[o[:8]].mean() - c[o[-8:]].mean())


def signal_analysis(qs, split_of, vlm: bool) -> dict:
    """C23 on held-out questions at N = 32 (calibration over all 32 rollouts of the question)."""
    held = [q for q in qs if split_of[q["sid"]] == "heldout"]
    names = ["L", "K"] + (["V", "Z", "Ztok"] if vlm else [])
    spear = {s: [] for s in names}
    gaps = {s: [] for s in names + ["length"]}
    corr_cv, cov_err, traj = [], 0.0, []
    for q in held:
        sc = {"L": finite(q["L"]), "K": finite(q["K"])}
        if vlm:
            z, zt = visconf(q["kl"], q["v"])
            sc |= {"V": finite(q["V"]), "Z": finite(z), "Ztok": finite(zt)}
            cs = calibrated_conf(q["kl"])
            cov = np.array([np.mean(c * v) - c.mean() * v.mean() if len(c) else 0.0 for c, v in zip(cs, q["v"])])
            ok = np.isfinite(z) & np.isfinite(zt)
            cov_err = max(cov_err, float(np.abs((zt - z)[ok] - cov[ok]).max()) if ok.any() else 0.0)
            per = [pearsonr(c, v)[0] for c, v in zip(cs, q["v"]) if len(c) > 2 and c.std() > 0 and v.std() > 0]
            if per:
                corr_cv.append(float(np.mean(per)))
        T = np.asarray(q["T"], dtype=float)
        for s in names:
            x = sc[s]
            ok = np.isfinite(x)
            if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(T[ok]) > 0:
                spear[s].append(float(spearmanr(T[ok], x[ok])[0]))
            gaps[s].append(quartile_gap(x, q["correct"]))
        gaps["length"].append(quartile_gap(T, q["correct"]))
    out = {"questions": len(held),
           "spearman_length_vs_score": {s: {"mean": float(np.mean(v)), "defined": len(v), "excluded": len(held) - len(v)}
                                        for s, v in spear.items()},
           "quartile_gap_correctness": {s: float(np.mean(v)) for s, v in gaps.items()}}
    if vlm:
        out["conf_vision_corr_per_question"] = {"mean": float(np.mean(corr_cv)), "median": float(np.median(corr_cv)),
                                                "q25": float(np.percentile(corr_cv, 25)),
                                                "q75": float(np.percentile(corr_cv, 75)), "questions": len(corr_cv)}
        out["cov_identity_max_abs_err"] = cov_err
    return out


def trajectories(qs, split, orders, n=3) -> list[dict]:
    """Fixed rule: first 3 held-out MathVista questions (permutation order), first rollout of order 0."""
    by_id = {q["sid"]: q for q in qs}
    out = []
    for sid in split["heldout"][:n]:
        q = by_id[sid]
        rid = orders[sid][0][0]
        c = calibrated_conf(q["kl"])[rid]
        out.append({"sid": sid, "rollout": rid, "correct": q["correct"][rid], "c": c.tolist(), "v": q["v"][rid].tolist()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista"], required=True)
    args = ap.parse_args()
    ds, vlm = args.dataset, args.dataset == "mathvista"
    with open(ROOT / f"data/cache/task3/{ds}.pkl", "rb") as f:
        qs = pickle.load(f)
    split = read_json(ROOT / f"data/task3_split_{ds}.json")
    orders = read_json(ROOT / f"data/task3_orders_{ds}.json")
    split_of = {s: "dev" for s in split["dev"]} | {s: "heldout" for s in split["heldout"]}
    eq = equiv_fn(ds)
    out = ROOT / f"results/task3/selection/{ds}_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for q in qs:
            for r in question_rows(q, orders[q["sid"]], split_of[q["sid"]], eq, vlm):
                fh.write(json.dumps(r) + "\n")
    rows = read_jsonl(out)  # summaries come from the saved rows, never from running counters
    write_json(ROOT / f"results/task3/selection/{ds}_summary.json", summarize(rows))
    pairs = [("sc", "first"), ("sc_rwc", "sc"), ("sc_high", "sc"), ("logp_high", "sc"), ("shortest", "sc")]
    if vlm:
        pairs += [("visconf_rwc", "sc"), ("visconf_rwc", "sc_rwc"), ("visconf_rwc", "vis_rwc"),
                  ("visconf_rwc", "visconf_high"), ("tokfuse_rwc", "visconf_rwc"), ("tokfuse_high", "visconf_high")]
    write_json(ROOT / f"results/task3/selection/{ds}_bootstrap.json", bootstrap(rows, pairs))
    sig = signal_analysis(qs, split_of, vlm)
    if vlm:
        sig["trajectories"] = trajectories(qs, split, orders)
    write_json(ROOT / f"results/task3/signals/{ds}.json", sig)
    s = summarize(rows)
    for m in BASE_METHODS + (VLM_METHODS if vlm else []):
        print(m, [round(s[f"heldout|{n}|{m}"]["acc"], 3) for n in BUDGETS])
    print("oracle", [round(s[f"heldout|{n}|sc"]["oracle"], 3) for n in BUDGETS])
    print(json.dumps({k: v for k, v in sig.items() if k != "trajectories"}, indent=1)[:2500])


if __name__ == "__main__":
    main()
