"""C24: single long response vs three parallel samples vs sequential refinement on 50 MATH questions.

python -m task3.live --mode {smoke,full} --store hub:<repo>
Qwen2.5-3B-Instruct at the README's revision, FP16, batch 1, T 0.7, top-p 0.8, top-k 20, repetition penalty 1.0.
Calls per question (seed = 6304 + 1000 * position + call, D19): 0 single (3,072), 1-3 parallel (1,024 each),
4 revision (2,048) conditioned on parallel response 1. Gold is used only to grade, never to choose.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.results import ROOT, read_json, write_json
from common.seeding import SEED
from task3.answers import math_equiv, math_key
from task3.selection import NO_ANSWER, greedy_groups, self_consistency

MODEL, REV = "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PROBLEMS = ROOT / "data/cache/task3_live_problems.json"
SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVISE = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}


def last_boxed(text: str):
    """README rule: last literal \\boxed{, balanced braces, stripped; absent, empty or unclosed -> None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    if depth:
        return None
    ans = text[start:j - 1].strip()
    return ans or None


def call(model, tok, messages, cap, seed, device):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=cap, pad_token_id=tok.eos_token_id, **SAMPLING)
    if device.type == "cuda":
        torch.cuda.synchronize()
    new = out[0, ids["input_ids"].shape[1]:].tolist()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    ended = any(t in eos for t in new)
    text = tok.decode(new, skip_special_tokens=True)
    return {"seed": seed, "cap": cap, "input_tokens": int(ids["input_ids"].shape[1]),
            "output_tokens": len([t for t in new if t not in eos]) + int(ended), "seconds": time.time() - t0,
            "truncated": len(new) >= cap and not ended, "response": text, "answer": last_boxed(text)}


def grade(ans, gold) -> bool:
    k = math_key(ans)
    return k != NO_ANSWER and math_equiv(k, math_key(gold))


def prepare() -> None:
    """The 50 problems and gold answers, read from the supplied export so grading matches the offline task."""
    import pyarrow.parquet as pq
    ids = set(read_json(ROOT / "data/task3_live_ids.json")["ids"])
    f = pq.ParquetFile(ROOT / "data/cache/task3_raw/qwen25-math/generations.parquet")
    out = {}
    for g in range(f.metadata.num_row_groups):
        r = f.read_row_group(g, columns=["sample_id", "question_text", "gold_answer"]).slice(0, 1).to_pylist()[0]
        if r["sample_id"] in ids:
            out[r["sample_id"]] = {"problem": r["question_text"], "gold": r["gold_answer"]}
    assert len(out) == 50
    write_json(PROBLEMS, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--store", default="")
    args = ap.parse_args()
    if args.prepare:
        return prepare()
    smoke = args.mode == "smoke"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = read_json(ROOT / "data/task3_live_ids.json")["ids"][: 2 if smoke else 50]
    problems = read_json(PROBLEMS)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct" if smoke else MODEL,
                                                 revision=None if smoke else REV, dtype=torch.float16).to(device).eval()
    caps = {"single": 64, "parallel": 32, "revision": 48} if smoke else {"single": 3072, "parallel": 1024, "revision": 2048}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = ROOT / f"results/task3/live{'_smoke' if smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "calls.jsonl", "w") as fh:
        for pos, sid in enumerate(ids):
            problem, gold_ans = problems[sid]["problem"], problems[sid]["gold"]
            user = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{problem}\n\n{SUFFIX}"}]
            seed = lambda c: SEED + 1000 * pos + c
            calls = {"single": call(model, tok, user, caps["single"], seed(0), device)}
            for k in (1, 2, 3):
                calls[f"parallel_{k}"] = call(model, tok, user, caps["parallel"], seed(k), device)
            init = calls["parallel_1"]
            revise_msgs = user + [{"role": "assistant", "content": init["response"]}, {"role": "user", "content": REVISE}]
            calls["revision"] = call(model, tok, revise_msgs, caps["revision"], seed(4), device)
            for name, c in calls.items():
                fh.write(json.dumps({"sid": sid, "position": pos, "call": name, "gold": gold_ans,
                                     "correct": grade(c["answer"], gold_ans), **c}) + "\n")
            keys = [math_key(calls[f"parallel_{k}"]["answer"]) for k in (1, 2, 3)]
            sc = self_consistency(greedy_groups(keys, math_equiv))
            rows.append({"sid": sid, "position": pos, "gold": gold_ans,
                         "single_correct": grade(calls["single"]["answer"], gold_ans),
                         "parallel_answer": calls[f"parallel_{sc + 1}"]["answer"],
                         "parallel_correct": grade(calls[f"parallel_{sc + 1}"]["answer"], gold_ans),
                         "initial_correct": grade(init["answer"], gold_ans),
                         "revision_correct": grade(calls["revision"]["answer"], gold_ans)})
            fh.flush()
            print(pos, sid, {k: v for k, v in rows[-1].items() if k.endswith("correct")}, flush=True)
    with open(out / "questions.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    write_json(out / "run_info.json", {"model": MODEL, "revision": REV, "dtype": "float16", "batch_size": 1,
                                       "sampling": SAMPLING, "caps": caps, "seed_rule": "6304 + 1000*position + call",
                                       "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                       "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
    if not smoke:
        from common.hub_sync import make_store
        store = make_store(args.store)
        for p in out.iterdir():
            store.put(p, f"task3/live/{p.name}")


if __name__ == "__main__":
    main()
