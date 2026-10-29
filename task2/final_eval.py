"""C17: final evaluation of both test suites under four image conditions.

python -m task2.final_eval --mode {smoke,full} --store hub:<repo>
The only entry point that reads the final-test suites (assert_no_test_in_selection). The model is the Stage-3
end, never a selected checkpoint. Mismatched map: fixed with seed 6304 and saved before generation; each item
gets an image from another pair of the same suite (assert_mismatch_map_fixed).
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

from common.guards import GuardError
from common.results import ROOT, read_json, write_json
from common.seeding import SEED
from task2.vlm import collate, user_text
from task2.vqa_norm import normalize, soft_accuracy

CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
GEN_CAP, BATCH, BOOT = 16, 8, 4000


def assert_no_test_in_selection():
    main = getattr(sys.modules.get("__main__"), "__spec__", None)
    if not (main and main.name == "task2.final_eval") and "pytest" not in sys.modules:
        raise GuardError("final-test suites may only be loaded by task2.final_eval")


@torch.no_grad()
def generate(tr, exs, image_of, image: bool) -> list[dict]:
    """Greedy, thinking off, 16 new tokens, batch 8, left padding."""
    tr.model.lm.eval()
    eot = tr.tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for m in range(0, len(exs), BATCH):
        chunk = exs[m:m + BATCH]
        seqs = [tr.model.enc.example(user_text(x["kind"], x.get("question")), None, image) for x in chunk]
        fs = [tr.feats[str(image_of(x))] if image else None for x in chunk]
        ids, attn, _ = collate(seqs, tr.tok.pad_token_id, tr.device, left=True)
        sync = torch.cuda.synchronize if tr.device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.time()
        with torch.autocast(tr.device.type, dtype=torch.float16, enabled=tr.amp):
            emb = tr.model.embed(ids, fs)
            gen = tr.model.lm.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=GEN_CAP, do_sample=False,
                                       temperature=None, top_p=None, top_k=None, use_cache=True,
                                       pad_token_id=tr.tok.pad_token_id, eos_token_id=eot)
        sync()
        dt = (time.time() - t0) / len(chunk)
        for x, g in zip(chunk, gen):
            g = g.tolist()
            text = tr.tok.decode([t for t in g if t not in (eot, tr.tok.pad_token_id)], skip_special_tokens=True).strip()
            out.append({"pred": text, "empty": text == "", "capped": len(g) >= GEN_CAP and eot not in g, "seconds": dt})
    return out


def mismatch_map(pairs, rng) -> dict:
    """item key -> image id from a different pair of the same suite."""
    keys = [(p["pair_id"], k) for p in pairs for k in range(2)]
    out = {}
    for i, p in enumerate(pairs):
        own = {x["image_id"] for x in p["items"]}
        others = [x["image_id"] for j, q in enumerate(pairs) if j != i for x in q["items"] if x["image_id"] not in own]
        for k in range(2):
            out[f"{p['pair_id']}|{k}"] = int(rng.choice(others))
    assert len(out) == len(keys)
    return out


def assert_mismatch_map_fixed(path, pairs):
    if not path.exists():
        raise GuardError("mismatched-image map must exist before generation")
    mm = read_json(path)
    for p in pairs:
        own = {x["image_id"] for x in p["items"]}
        if any(mm[f"{p['pair_id']}|{k}"] in own for k in range(2)):
            raise GuardError(f"mismatched image belongs to its own pair: {p['pair_id']}")


def score(x, pred: str) -> float:
    return soft_accuracy(pred, x["answers"]) if x["kind"] == "vqa" else float(normalize(pred) == normalize(x["target"]))


def canonical(x) -> str:
    return x["consensus"] if x["kind"] == "vqa" else normalize(x["target"])


def evaluate_suite(tr, suite: str, pairs, mm) -> list[dict]:
    rows = []
    exs = [x for p in pairs for x in p["items"]]
    keys = [f"{p['pair_id']}|{k}" for p in pairs for k in range(2)]
    pair_of = [p for p in pairs for _ in range(2)]
    partner = [p["items"][1 - k]["image_id"] for p in pairs for k in range(2)]
    image_for = {"correct": lambda i: exs[i]["image_id"], "complementary": lambda i: partner[i],
                 "mismatched": lambda i: mm[keys[i]], "no_image": None}
    for cond in CONDITIONS:
        idx = {id(x): i for i, x in enumerate(exs)}
        f = image_for[cond]
        img = cond != "no_image"
        gens = generate(tr, exs, (lambda x: f(idx[id(x)])) if img else None, img)
        nll = tr.nll_of(exs, image_of=(lambda x: f(idx[id(x)])) if img else None, image=img)
        for i, (x, g) in enumerate(zip(exs, gens)):
            other = pair_of[i]["items"][1 - i % 2]
            rows.append({"suite": suite, "condition": cond, "pair_id": pair_of[i]["pair_id"], "slot": i % 2,
                         "type": pair_of[i]["type"], "question": x["question"], "image_id": x["image_id"],
                         "image_used": f(i) if img else None, "reference": x["target"], "answers": x["answers"],
                         "pred": g["pred"], "score": score(x, g["pred"]), "match_own": normalize(g["pred"]) == canonical(x),
                         "match_partner": normalize(g["pred"]) == canonical(other), "empty": g["empty"],
                         "capped": g["capped"], "nll": nll[i], "seconds": g["seconds"]})
    return rows


def summarize(rows) -> dict:
    out = {}
    for suite in sorted({r["suite"] for r in rows}):
        for cond in CONDITIONS:
            rs = [r for r in rows if r["suite"] == suite and r["condition"] == cond]
            pairs = {}
            for r in rs:
                pairs.setdefault(r["pair_id"], []).append(r)
            d = {"n": len(rs), "accuracy": float(np.mean([r["score"] for r in rs])),
                 "by_type": {t: float(np.mean([r["score"] for r in rs if r["type"] == t])) for t in sorted({r["type"] for r in rs})},
                 "pair_accuracy": float(np.mean([all(r["match_own"] for r in v) for v in pairs.values()])),
                 "identical_answer_rate": float(np.mean([normalize(v[0]["pred"]) == normalize(v[1]["pred"]) for v in pairs.values()])),
                 "empty_rate": float(np.mean([r["empty"] for r in rs])),
                 "mean_reference_nll": float(np.mean([r["nll"] for r in rs])),
                 "seconds_per_answer": float(np.mean([r["seconds"] for r in rs]))}
            if cond == "complementary":
                d["follows_partner"] = float(np.mean([r["match_partner"] for r in rs]))
                d["matches_own"] = float(np.mean([r["match_own"] for r in rs]))
                d["neither"] = float(np.mean([not r["match_partner"] and not r["match_own"] for r in rs]))
            out[f"{suite}|{cond}"] = d
    return out


def advantages(rows) -> dict:
    """Correct minus complementary accuracy, complementary minus correct NLL; 4,000 resamples of complete pairs."""
    rng = np.random.default_rng(SEED)
    out = {}
    for suite in sorted({r["suite"] for r in rows}):
        pid = sorted({r["pair_id"] for r in rows if r["suite"] == suite})
        def per_pair(cond, field):
            m = {}
            for r in rows:
                if r["suite"] == suite and r["condition"] == cond:
                    m.setdefault(r["pair_id"], []).append(float(r[field]))
            return np.array([np.mean(m[p]) for p in pid])
        draws = rng.integers(0, len(pid), size=(BOOT, len(pid)))   # same pair draws for both conditions
        for name, a, b, field in [("accuracy", "correct", "complementary", "score"),
                                  ("nll", "complementary", "correct", "nll"),
                                  ("accuracy_vs_no_image", "correct", "no_image", "score"),
                                  ("accuracy_vs_mismatched", "correct", "mismatched", "score")]:
            d = per_pair(a, field) - per_pair(b, field)
            boots = d[draws].mean(1)
            out[f"{suite}|{name}"] = {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
                                      "hi": float(np.percentile(boots, 97.5)), "pairs": len(pid)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    assert_no_test_in_selection()
    from task2.train import Trainer, load_cfg, load_inputs
    cfg = load_cfg(args.mode)
    items, feats = load_inputs(args.mode)
    sfx = "_smoke" if args.mode == "smoke" else ""
    tr = Trainer(cfg, items, feats, args.store, ROOT / f"results/task2/train{sfx}")
    tr.attach_lora()
    w = tr.store.get(f"{tr.run}/stage3/weights.pt", ROOT / f"results/task2/train{sfx}/stage3_weights_eval.pt")
    st = torch.load(w, map_location=tr.device, weights_only=False)
    tr.model.connector.load_state_dict(st["connector"])
    with torch.no_grad():
        for n, p in tr.model.lm.named_parameters():
            if ".lora_" in n:
                p.copy_(st["lora"][n])
    out = ROOT / f"results/task2/final{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    suites = {"grounding": items["grounding_test"], "vqa": items["vqa_test"]}
    for name, pairs in suites.items():
        path = out / f"mismatch_map_{name}.json"
        if not path.exists():
            write_json(path, mismatch_map(pairs, rng))
        assert_mismatch_map_fixed(path, pairs)
    if tr.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for name, pairs in suites.items():
        rows += evaluate_suite(tr, name, pairs, read_json(out / f"mismatch_map_{name}.json"))
    with open(out / "rows.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    rows = [json.loads(l) for l in open(out / "rows.jsonl")]
    summary = summarize(rows)
    summary["peak_memory_bytes"] = torch.cuda.max_memory_allocated() if tr.device.type == "cuda" else None
    write_json(out / "summary.json", summary)
    write_json(out / "advantages.json", advantages(rows))
    for p in out.iterdir():
        tr.store.put(p, f"{tr.run}/final/{p.name}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("accuracy", "pair_accuracy")} for k, v in summary.items()
                      if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
