"""C20: join the supplied tables, verify alignment, normalize answers, label correctness, split and order.

python -m task3.load --dataset {math,mathvista} [--mode smoke]
Writes data/task3_split_<ds>.json, data/task3_orders_<ds>.json (committed) and data/cache/task3/<ds>.pkl.
Correctness labels live in their own array (`correct`), never in the fields selection reads.
"""
import argparse
import json
import pickle
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from common.results import ROOT, write_json
from common.seeding import SEED
from task3.answers import SCORER_VERSION, math_equiv, math_key, mathvista_key
from task3.selection import NO_ANSWER, greedy_groups
from task3.signals import mean_logp, self_certainty, visual_engagement_tokens

RAW = ROOT / "data/cache/task3_raw"
FOLDER = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
N_DEV, N_ORDERS, R = 200, 10, 32


def equiv_fn(ds):
    return math_equiv if ds == "math" else (lambda a, b: a == b)


def load_question(ds, files, g: int, meta) -> dict:
    gen = files["gen"].read_row_group(g).to_pylist()
    prob = files["prob"].read_row_group(g).to_pylist()
    att = files["att"].read_row_group(g).to_pylist() if "att" in files else None
    key = lambda r: (r["dataset"], r["sample_id"], r["rollout_id"])
    gen, prob = sorted(gen, key=key), sorted(prob, key=key)
    assert [key(r) for r in gen] == [key(r) for r in prob] and len(gen) == R
    if att is not None:
        att = sorted(att, key=key)
        assert [key(r) for r in att] == [key(r) for r in gen]
    sid = gen[0]["sample_id"]
    assert all(r["sample_id"] == sid for r in gen) and [r["rollout_id"] for r in gen] == list(range(R))
    q = {"sid": sid, "T": [], "hit_max": [], "raw": [], "keys": [], "L": [], "K": [], "kl": [], "V": [], "v": []}
    for i, (rg, rp) in enumerate(zip(gen, prob)):
        n = rg["n_generated_text_tokens"]
        lens = {len(rg["generated_token_ids"]), len(rp["logp"]), len(rp["kl_u_p"])}
        if att is not None:
            ra = att[i]
            lens |= {len(ra["img_attn_total"]), len(ra["prompt_text_attn_total"]), len(ra["generated_text_attn_total"])}
        assert lens == {n}, f"token misalignment {sid}/{i}: {lens} vs {n}"
        q["T"].append(n)
        q["hit_max"].append(rg["hit_max_new_tokens"])
        q["raw"].append(rg["extracted_answer"])
        q["keys"].append(math_key(rg["extracted_answer"]) if ds == "math" else mathvista_key(rg["extracted_answer"], meta[sid]))
        kl = np.asarray(rp["kl_u_p"], dtype=np.float32)
        q["L"].append(mean_logp(np.asarray(rp["logp"])) if n else float("nan"))
        q["K"].append(self_certainty(kl) if n else float("nan"))
        q["kl"].append(kl)
        if att is not None:
            v = visual_engagement_tokens(ra["img_attn_total"], ra["prompt_text_attn_total"], ra["generated_text_attn_total"])
            q["v"].append(v.astype(np.float32))
            q["V"].append(float(v.mean()) if n else float("nan"))
    gold = gen[0]["gold_answer"]
    q["gold_key"] = math_key(gold) if ds == "math" else mathvista_key(gold, meta[sid])
    eq = equiv_fn(ds)
    q["correct"] = [k != NO_ANSWER and not k.startswith(("UNMATCHED:", "UNPARSED:", "NONINT:")) and eq(k, q["gold_key"])
                    for k in q["keys"]]
    return q


def split_and_orders(ids: list[str]) -> tuple[dict, dict]:
    perm = list(np.random.default_rng(SEED).permutation(sorted(ids)))
    split = {"seed": SEED, "rule": "sorted ids, numpy default_rng(6304).permutation; first 200 dev",
             "order": perm, "dev": perm[:N_DEV], "heldout": perm[N_DEV:]}
    rng = np.random.default_rng(SEED)
    orders = {sid: [rng.permutation(R).tolist() for _ in range(N_ORDERS)] for sid in sorted(ids)}
    return split, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(FOLDER), required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    ds = args.dataset
    base = RAW / FOLDER[ds]
    files = {"gen": pq.ParquetFile(base / "generations.parquet"), "prob": pq.ParquetFile(base / "probability_metrics.parquet")}
    if ds == "mathvista":
        files["att"] = pq.ParquetFile(base / "attention_metrics.parquet")
    meta = json.loads((ROOT / "data/cache/mathvista_meta/testmini.json").read_text()) if ds == "mathvista" else None
    n_groups = files["gen"].metadata.num_row_groups if args.mode == "full" else 5
    qs = [load_question(ds, files, g, meta) for g in range(n_groups)]

    ids = [q["sid"] for q in qs]
    assert len(set(ids)) == len(ids)
    gold_bad = [q["sid"] for q in qs if q["gold_key"] == NO_ANSWER or q["gold_key"].startswith(("UNMATCHED:", "UNPARSED:", "NONINT:"))]
    report = {"dataset": ds, "questions": len(qs), "rollouts": sum(len(q["T"]) for q in qs),
              "scorer_version": SCORER_VERSION, "gold_not_normalized": gold_bad,
              "no_answer": sum(k == NO_ANSWER for q in qs for k in q["keys"]),
              "capped": sum(sum(q["hit_max"]) for q in qs), "retained_tokens": int(sum(sum(q["T"]) for q in qs)),
              "rollout_accuracy": float(np.mean([c for q in qs for c in q["correct"]]))}
    if ds == "mathvista":
        mc = [k for q in qs if meta[q["sid"]]["question_type"] == "multi_choice" for k in q["keys"] if k != NO_ANSWER]
        report["mc_unmatched_rate"] = sum(k.startswith("UNMATCHED:") for k in mc) / max(len(mc), 1)
        report["mc_unmatched_examples"] = Counter(k for k in mc if k.startswith("UNMATCHED:")).most_common(20)
        report["free_form_unparsed_rate"] = float(np.mean([k.startswith(("UNPARSED:", "NONINT:")) for q in qs
                                                           if meta[q["sid"]]["question_type"] == "free_form"
                                                           for k in q["keys"] if k != NO_ANSWER]))
    # D16 gate: within one full prefix, groups whose first members are equivalent (non-transitive merges)
    eq = equiv_fn(ds)
    split, orders = split_and_orders(ids)
    cross = 0
    for q in qs:
        keys = [q["keys"][i] for i in orders[q["sid"]][0]]
        gid = greedy_groups(keys, eq)
        reps = [keys[gid.index(g)] for g in sorted(set(gid))]
        cross += sum(eq(a, b) for i, a in enumerate(reps) for b in reps[i + 1:] if NO_ANSWER not in (a, b))
    report["equivalent_group_pairs_in_full_prefix"] = cross
    if args.mode == "full":
        write_json(ROOT / f"data/task3_split_{ds}.json", split)
        write_json(ROOT / f"data/task3_orders_{ds}.json", orders)
        write_json(ROOT / f"results/task3/load_report_{ds}.json", report)
        if ds == "math":
            write_json(ROOT / "data/task3_live_ids.json", {"rule": "MATH permutation positions 200-249 (S10)",
                                                         "ids": split["order"][200:250]})
    out = ROOT / f"data/cache/task3/{ds}{'_smoke' if args.mode == 'smoke' else ''}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(qs, f)
    print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main()
