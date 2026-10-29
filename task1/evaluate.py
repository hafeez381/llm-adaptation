"""C7 + C9: evaluate one condition on GSM8K / HellaSwag / BoolQ / ARC-Easy and generate the 24 qualitative responses.

python -m task1.evaluate --condition <c> --store hub:<repo> --mode {smoke,full}
Settings are identical for every condition (D4): lm-eval HF backend, FP16, batch 32, zero-shot.
GSM8K: gsm8k_cot, num_fewshot=0, non-thinking chat template, greedy, 1,024 new tokens, stock filters.
Writes results/task1/eval/<c>/ (aggregates, per-example rows, eval config) and results/task1/qual/<c>.jsonl.
"""
import argparse
import json
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.guards import GuardError, assert_probe_frozen
from common.hub_sync import make_store
from common.results import ROOT, read_json, write_json
from common.seeding import SEED
from task1.merge import CONDITIONS, resolve

GSM_CAP, QUAL_CAP, BATCH = 1024, 512, 32
LOGLIK = ["hellaswag", "boolq", "arc_easy"]
EXPECTED = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}


def assert_zero_shot(cfg: dict) -> None:
    if any(v != 0 for v in cfg["num_fewshot"].values()) or not cfg["gsm8k_chat_template"]:
        raise GuardError(f"evaluation is not zero-shot with the GSM8K chat template: {cfg}")


def run_lm_eval(path, tasks, chat: bool, limit, gen_kwargs=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=str(path), dtype="float16", batch_size=BATCH, enable_thinking=False, max_length=4096,
              device="cuda" if torch.cuda.is_available() else "cpu")
    return lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, apply_chat_template=chat,
                                   gen_kwargs=gen_kwargs, log_samples=True, limit=limit, random_seed=SEED,
                                   numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED,
                                   bootstrap_iters=0)


def gsm8k_rows(res, tok, cap: int) -> list[dict]:
    """One row per document; lm-eval logs one sample record per answer filter, merged here."""
    rows = {}
    for s in res["samples"]["gsm8k_cot"]:
        d = s["doc_id"]
        if d not in rows:
            resp = s["resps"][0][0]
            n_tok = len(tok(resp, add_special_tokens=False).input_ids)
            rows[d] = {"doc_id": d, "prompt": s["arguments"][0][0] if s.get("arguments") else None,
                       "response": resp, "response_tokens": n_tok,
                       # S5: a stock-stopped response that re-tokenizes to the cap never reached end-of-turn
                       "truncated": n_tok >= cap - 1, "target": s["target"]}
        f = s["filter"].replace("-", "_")
        ext = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        rows[d] |= {f"exact_match_{f}": s["exact_match"], f"extracted_{f}": ext}
    return [rows[d] for d in sorted(rows)]


def qual_generate(path, tok, limit, cap=QUAL_CAP) -> list[dict]:
    """C9: one greedy response per prompt, thinking off, 512 new tokens."""
    entries = yaml.safe_load((ROOT / "data/cache/first_plot_questions.yaml").read_text())
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(
        "cuda" if torch.cuda.is_available() else "cpu").eval()
    rows = []
    for e in entries[:limit]:
        msgs = ([{"role": "system", "content": e["system"]}] if "system" in e else []) + \
               [{"role": "user", "content": e["paraphrases"][0]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=cap, do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
        new = out[0, ids["input_ids"].shape[1]:]
        eot = tok.convert_tokens_to_ids("<|im_end|>")
        base = e["id"].removesuffix("_json").removesuffix("_template")
        fmt = "json" if e["id"].endswith("_json") else "template" if e["id"].endswith("_template") else "plain"
        rows.append({"prompt_id": e["id"], "question": base, "format": fmt,
                     "response": tok.decode(new, skip_special_tokens=True), "new_tokens": int(len(new)),
                     "truncated": bool(len(new) >= cap and eot not in new.tolist())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    limit = 4 if smoke else None
    man = read_json(ROOT / "data/manifest.json")
    out_dir = ROOT / f"results/task1/eval/{args.condition}"
    assert_probe_frozen(ROOT / "data/probe_ids.json", man, list(out_dir.glob("*.json")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = make_store(args.store)
    path = resolve(args.condition, store, ROOT / "outputs/models", device)
    tok = AutoTokenizer.from_pretrained(path)
    cfg = {"condition": args.condition, "backend": "lm-eval hf", "dtype": "float16", "batch_size": BATCH,
           "num_fewshot": {t: 0 for t in EXPECTED}, "gsm8k_chat_template": True, "enable_thinking": False,
           "gsm8k_gen": {"do_sample": False, "max_gen_toks": GSM_CAP}, "loglik_chat_template": False,
           "qual": {"do_sample": False, "max_new_tokens": QUAL_CAP}, "seed": SEED, "mode": args.mode}
    assert_zero_shot(cfg)

    gsm_cap = 64 if smoke else GSM_CAP  # smoke only checks plumbing on the CPU
    t0 = time.time()
    g = run_lm_eval(path, ["gsm8k_cot"], True, limit, f"do_sample=False,max_gen_toks={gsm_cap}")
    t_gsm = time.time() - t0
    t0 = time.time()
    ll = run_lm_eval(path, LOGLIK, False, limit)
    t_ll = time.time() - t0
    rows = gsm8k_rows(g, tok, gsm_cap)
    agg = {"gsm8k_cot": g["results"]["gsm8k_cot"]} | {t: ll["results"][t] for t in LOGLIK}
    if not smoke:
        n = {"gsm8k_cot": len(rows)} | {t: len(ll["samples"][t]) for t in LOGLIK}
        assert n == EXPECTED, n
    agg["gsm8k_truncation_rate"] = sum(r["truncated"] for r in rows) / len(rows)
    agg["seconds"] = {"gsm8k": t_gsm, "loglik": t_ll}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gsm8k_samples.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    for t in LOGLIK:
        with open(out_dir / f"{t}_samples.jsonl", "w") as f:
            f.writelines(json.dumps({"doc_id": s["doc_id"], "target": s["target"], "acc": s.get("acc"),
                                     "acc_norm": s.get("acc_norm")}) + "\n" for s in ll["samples"][t])
    write_json(out_dir / "aggregates.json", agg)
    write_json(out_dir / "eval_config.json", cfg)

    qual = qual_generate(path, tok, 2 if smoke else None, 16 if smoke else QUAL_CAP)
    qdir = ROOT / "results/task1/qual"
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / f"{args.condition}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in qual)
    if not smoke:
        for p in [*out_dir.iterdir(), qdir / f"{args.condition}.jsonl"]:
            store.put(p, f"task1/eval/{args.condition}/{p.name}")
    print(json.dumps({k: v for k, v in agg.items() if k != "seconds"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
