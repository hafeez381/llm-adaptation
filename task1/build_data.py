"""C1: the first 40,000 eligible MetaMathQA examples in source order, tokenized with masks.

python -m task1.build_data --mode {smoke,full}
Writes data/cache/task1_train[_smoke].pt (not committed) and, in full mode, data/task1_train_ids.json
plus the `task1_data` block of data/manifest.json. The VM reruns this script and the trainer checks
the ordered-ID sha256 against the committed manifest, so both machines train on identical data.
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from common.guards import assert_no_gsm8k_overlap, assert_source_order, normalize_text
from common.results import ROOT, read_json, sha256_json, write_json

MODEL = "Qwen/Qwen3-0.6B"
N_EXAMPLES = 40_000
MAX_LEN = 2048


def prompt_ids(tok, messages: list[dict]) -> list[int]:
    """Chat prompt with thinking off; ends with the assistant header and an empty think block."""
    return tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                   tokenize=True, return_dict=False)


def format_example(tok, query: str, response: str) -> tuple[list[int], int]:
    """Returns (token ids, prompt length). Tokens from prompt_len on are the response plus <|im_end|>."""
    p = prompt_ids(tok, [{"role": "user", "content": query}])
    r = tok(response, add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return p + r, len(p)


def build(tok, n: int, max_len: int, gsm8k_norm: set[str]):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    ids, seqs, plens, excluded = [], [], [], {"empty": 0, "too_long": 0, "gsm8k_match": 0}
    for i, ex in enumerate(ds):
        if len(ids) == n:
            break
        q, r = ex["query"], ex["response"]
        if not q.strip() or not r.strip():
            excluded["empty"] += 1
            continue
        if normalize_text(q) in gsm8k_norm or normalize_text(ex["original_question"]) in gsm8k_norm:
            excluded["gsm8k_match"] += 1
            continue
        seq, plen = format_example(tok, q, r)
        if len(seq) > max_len:
            excluded["too_long"] += 1
            continue
        ids.append(i), seqs.append(seq), plens.append(plen)
    assert len(ids) == n, f"only {len(ids)} eligible examples"
    questions = [ds[i]["query"] for i in ids] + [ds[i]["original_question"] for i in ids]
    return ids, seqs, plens, excluded, questions, len(ids) + sum(excluded.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    n, max_len = (64, 128) if smoke else (N_EXAMPLES, MAX_LEN)

    man = read_json(ROOT / "data/manifest.json")
    rev = man["revisions"]["hf_model:Qwen/Qwen3-0.6B"]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    gsm8k_norm = {normalize_text(x["question"]) for x in load_dataset("openai/gsm8k", "main", split="test")}

    ids, seqs, plens, excluded, questions, scanned = build(tok, n, max_len, gsm8k_norm)
    assert_no_gsm8k_overlap(questions, gsm8k_norm)
    valid = [len(s) - p for s, p in zip(seqs, plens)]

    out = ROOT / f"data/cache/task1_train{'_smoke' if smoke else ''}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"source_ids": ids, "input_ids": seqs, "prompt_len": plens, "max_len": max_len}, out)
    block = {"n": n, "max_len": max_len, "source_rows_scanned": scanned, "excluded": excluded,
             "normalization": "NFKC, casefold, whitespace collapsed to one space, strip (S1)",
             "fields_checked": ["query", "original_question"],
             "ordered_ids_sha256": sha256_json(ids), "valid_tokens": valid,
             "valid_tokens_total": sum(valid), "max_formatted_len": max(map(len, seqs))}
    if not smoke:
        write_json(ROOT / "data/task1_train_ids.json", ids)
        man["task1_data"] = block
        write_json(ROOT / "data/manifest.json", man)
        assert_source_order(ids, block, list(map(len, seqs)), MAX_LEN)
    print({k: v for k, v in block.items() if k != "valid_tokens"})


if __name__ == "__main__":
    main()
