"""C9 (D7): summarize the 42 annotated responses and pick aligned excerpts by a fixed rule.

python -m task1.select_excerpts
Rule, fixed before reading any response: for each criterion, the (question, format) cell whose seven condition
flags are most split (largest min(#yes, #no)); ties go to the earlier question, then plain < json < template.
"""
import csv
import json

from common.results import ROOT, read_json, write_json
from task1.merge import CONDITIONS

CRITERIA = ["format_violation", "incoherent", "repetition", "math_framing", "inappropriate", "truncated"]
FORMATS = ["plain", "json", "template"]


def load_annotations():
    with open(ROOT / "task1/annotations.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in CRITERIA:
            r[c] = int(r[c])
    return rows


def main():
    rows = load_annotations()
    assert len(rows) == 42, len(rows)
    order = read_json(ROOT / "data/manifest.json")["qual_prompts"]["annotated_questions"]
    summary = {cond: {c: sum(r[c] for r in rows if r["condition"] == cond) for c in CRITERIA} for cond in CONDITIONS}
    picks = {}
    for c in CRITERIA:
        best = None
        for qi, q in enumerate(order):
            for fi, fmt in enumerate(FORMATS):
                cell = [r for r in rows if r["question"] == q and r["format"] == fmt]
                yes = sum(r[c] for r in cell)
                split = min(yes, len(cell) - yes)
                key = (-split, qi, fi)
                if best is None or key < best[0]:
                    best = (key, q, fmt, cell)
        _, q, fmt, cell = best
        picks[c] = {"question": q, "format": fmt, "split": -best[0][0],
                    "flags": {r["condition"]: r[c] for r in cell},
                    "excerpts": {r["condition"]: r["excerpt"] for r in cell}}
    write_json(ROOT / "results/task1/qual/annotation_summary.json", {"per_condition": summary, "excerpts": picks})
    print(json.dumps(summary, indent=1))
    for c, p in picks.items():
        print(c, p["question"], p["format"], "split", p["split"], p["flags"])


if __name__ == "__main__":
    main()
