"""BoolQ answer bias per condition: how much of a BoolQ change is a shift in the yes/no prior.

python -m task1.boolq_bias  ->  results/task1/boolq_bias.json
Reads the saved per-example BoolQ rows (target 1 = yes) and reports the predicted-yes rate and the accuracy on
gold-yes and gold-no questions separately.
"""
import json

from common.results import ROOT, write_json
from task1.merge import CONDITIONS


def bias(rows: list[dict]) -> dict:
    pred = [r["target"] if r["acc"] == 1 else 1 - r["target"] for r in rows]
    yes = [r for r in rows if r["target"] == 1]
    no = [r for r in rows if r["target"] == 0]
    return {"pred_yes": sum(pred) / len(rows), "gold_yes": len(yes) / len(rows),
            "acc_gold_yes": sum(r["acc"] for r in yes) / len(yes), "acc_gold_no": sum(r["acc"] for r in no) / len(no)}


def main():
    out = {}
    for c in CONDITIONS:
        with open(ROOT / f"results/task1/eval/{c}/boolq_samples.jsonl") as f:
            out[c] = bias([json.loads(l) for l in f])
    write_json(ROOT / "results/task1/boolq_bias.json", out)


if __name__ == "__main__":
    main()
