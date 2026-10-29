"""C26 (D20): illustrative cases chosen by fixed rules, never by eye.

python -m task3.select_cases
- selection disagreement: held-out questions where self-consistency and the quality-weighted RWC rule (self-certainty
  for MATH, VisConf for MathVista) pick different answers at N = 32 in order 0; the 2 smallest question ids.
- revision: the first wrong->correct and the first correct->wrong transition in live-subset order.
- premature stopping: the first 2 incorrect early stops (held-out permutation order, order 0) of each controller
  where fixed-32 self-consistency was correct.
"""
import json
import pickle

from common.results import ROOT, read_json, read_jsonl, write_json

RWC = {"math": "sc_rwc", "mathvista": "visconf_rwc"}


def sid_key(s: str):
    return int(s.split(":")[1]) if ":" in s else int(s)


def text_of(ds: str, sid: str, rid: int, n=400) -> str:
    import pyarrow.parquet as pq
    folder = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}[ds]
    t = pq.read_table(ROOT / f"data/cache/task3_raw/{folder}/generations.parquet",
                      filters=[("sample_id", "=", sid), ("rollout_id", "=", rid)],
                      columns=["question_text", "generated_text"]).to_pylist()[0]
    return t["generated_text"][-n:]


def main():
    out = {}
    for ds in ("math", "mathvista"):
        with open(ROOT / f"data/cache/task3/{ds}.pkl", "rb") as f:
            qs = {q["sid"]: q for q in pickle.load(f)}
        rows = [r for r in read_jsonl(ROOT / f"results/task3/selection/{ds}_rows.jsonl") if r["split"] == "heldout" and r["N"] == 32]
        pick = {}
        for r in rows:
            if r["method"] in ("sc", RWC[ds]):
                pick.setdefault(r["sid"], {})[r["method"]] = r["selected"][0]
        dis = sorted((s for s, p in pick.items() if qs[s]["keys"][p["sc"]] != qs[s]["keys"][p[RWC[ds]]]), key=sid_key)
        cases = []
        for s in dis[:2]:
            q = qs[s]
            entry = {"sid": s, "gold": q["gold_key"], "n_disagreeing_questions": len(dis)}
            for m in ("sc", RWC[ds]):
                rid = pick[s][m]
                entry[m] = {"rollout": rid, "answer": q["keys"][rid], "correct": q["correct"][rid],
                            "support": sum(k == q["keys"][rid] for k in q["keys"]), "K": q["K"][rid],
                            "tail": text_of(ds, s, rid)}
            cases.append(entry)
        out[f"{ds}_selection_disagreement"] = cases

        split = read_json(ROOT / f"data/task3_split_{ds}.json")["heldout"]
        full = {}
        for i, x in enumerate(read_jsonl(ROOT / f"results/task3/adaptive/{ds}_heldout_always32_rows.jsonl")):
            full[(x["sid"], i % 10)] = x
        for ctrl in ("agreement", "quality"):
            rs = read_jsonl(ROOT / f"results/task3/adaptive/{ds}_heldout_{ctrl}_rows.jsonl")
            by = {(x["sid"], i % 10): x for i, x in enumerate(rs)}
            chosen = []
            for s in split:
                x = by[(s, 0)]
                if x["stop"] < 32 and not x["correct"] and full[(s, 0)]["correct"]:
                    q = qs[s]
                    chosen.append({"sid": s, "stop": x["stop"], "early_answer": q["keys"][x["rollout"]],
                                   "fixed32_answer": q["keys"][full[(s, 0)]["rollout"]], "gold": q["gold_key"],
                                   "pool_fraction_correct": sum(q["correct"]) / 32})
                if len(chosen) == 2:
                    break
            out[f"{ds}_premature_stop_{ctrl}"] = chosen

    live = ROOT / "results/task3/live/questions.jsonl"
    if live.exists():
        qrows = read_jsonl(live)
        calls = {(c["sid"], c["call"]): c for c in read_jsonl(ROOT / "results/task3/live/calls.jsonl")}
        rev = {}
        for r in sorted(qrows, key=lambda r: r["position"]):
            kind = ("wrong_to_correct" if not r["initial_correct"] and r["revision_correct"] else
                    "correct_to_wrong" if r["initial_correct"] and not r["revision_correct"] else None)
            if kind and kind not in rev:
                rev[kind] = {"sid": r["sid"], "position": r["position"], "gold": r["gold"],
                             "initial_answer": calls[(r["sid"], "parallel_1")]["answer"],
                             "revised_answer": calls[(r["sid"], "revision")]["answer"],
                             "revision_tail": calls[(r["sid"], "revision")]["response"][-600:]}
        out["live_revision"] = rev
    write_json(ROOT / "results/task3/cases.json", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
