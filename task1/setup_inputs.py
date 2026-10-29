"""C0: record the environment and every external input, check the model facts, draw the probe set.

Run once before any model is evaluated:  python -m task1.setup_inputs
The probe draw is deterministic (seed 6304), and the manifest stores the probe file's sha256.
"""
import argparse
import importlib.metadata as md
import platform
import subprocess

import numpy as np
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from common.results import ROOT, read_json, sha256_file, sha256_json, write_json
from common.seeding import SEED

MANIFEST = ROOT / "data/manifest.json"
PROBES = ROOT / "data/probe_ids.json"
YAML_URL = ("https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
            "{rev}/evaluation/first_plot_questions.yaml")

HF_MODELS = ["Qwen/Qwen3-0.6B", "google/siglip-base-patch16-224", "Qwen/Qwen2.5-3B-Instruct",
             "Qwen/Qwen2.5-VL-3B-Instruct"]
HF_DATASETS = ["meta-math/MetaMathQA", "openai/gsm8k", "Rowan/hellaswag", "google/boolq",
               "allenai/ai2_arc"]
PACKAGES = ["torch", "transformers", "peft", "datasets", "huggingface_hub", "lm_eval", "math-verify",
            "numpy", "pandas", "pyarrow"]


def load_manifest() -> dict:
    return read_json(MANIFEST) if MANIFEST.exists() else {}


def env_info() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "packages": {p: md.version(p) for p in PACKAGES}}


def revisions() -> dict:
    api = HfApi()
    out = {f"hf_model:{m}": api.model_info(m).sha for m in HF_MODELS}
    out |= {f"hf_dataset:{d}": api.dataset_info(d).sha for d in HF_DATASETS}
    rev = subprocess.check_output(["gh", "api", "repos/emergent-misalignment/emergent-misalignment/commits/main",
                                   "--jq", ".sha"], text=True).strip()
    out["github:emergent-misalignment"] = rev
    return out


def model_facts(rev: str) -> dict:
    cfg = read_json(hf_hub_download("Qwen/Qwen3-0.6B", "config.json", revision=rev))
    keys = ["hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
            "num_hidden_layers", "tie_word_embeddings", "vocab_size"]
    facts = {k: cfg[k] for k in keys}
    assert all(i < facts["num_hidden_layers"] for i in (0, 13, 27)), "diagnostic layers missing"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=rev)
    facts["vision_tokens"] = {t: tok.convert_tokens_to_ids(t)
                              for t in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]}
    assert tok.unk_token_id not in facts["vision_tokens"].values()
    facts["eot"] = {"token": "<|im_end|>", "id": tok.convert_tokens_to_ids("<|im_end|>")}
    return facts


def qual_prompts(rev: str) -> dict:
    import urllib.request
    path = ROOT / "data/cache/first_plot_questions.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YAML_URL.format(rev=rev), path)
    entries = yaml.safe_load(path.read_text())
    assert len(entries) == 24 and all(len(e["paraphrases"]) == 1 for e in entries)
    plain = [e["id"] for e in entries if not e["id"].endswith(("_json", "_template"))]
    assert len(plain) == 8
    assert {e["id"] for e in entries} == {q + s for q in plain for s in ("", "_json", "_template")}
    return {"sha256": sha256_file(path), "n_prompts": 24, "plain_order": plain,
            "annotated_questions": [plain[0], plain[-1]]}


def draw_probes() -> dict:
    """32 GSM8K test questions and 32 HellaSwag validation contexts, one rng, in that order."""
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    assert (len(gsm), len(hs)) == (1319, 10042)
    rng = np.random.default_rng(SEED)
    g_ids = sorted(rng.choice(len(gsm), 32, replace=False).tolist())
    h_ids = sorted(rng.choice(len(hs), 32, replace=False).tolist())
    return {"seed": SEED, "gsm8k_test": g_ids, "hellaswag_validation": h_ids,
            "gsm8k_text_sha256": sha256_json([gsm[i]["question"] for i in g_ids]),
            "hellaswag_text_sha256": sha256_json([hs[i]["ctx"] for i in h_ids])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw-probes", action="store_true", help="refuse by default: probes are drawn once")
    args = ap.parse_args()

    write_json(ROOT / "results/env.json", env_info())
    man = load_manifest()
    man["revisions"] = revisions()
    man["model_facts"] = model_facts(man["revisions"]["hf_model:Qwen/Qwen3-0.6B"])
    man["qual_prompts"] = qual_prompts(man["revisions"]["github:emergent-misalignment"])
    if PROBES.exists() and not args.redraw_probes:
        print("probe file exists, kept unchanged")
    else:
        write_json(PROBES, draw_probes())
    man["probe_ids_sha256"] = sha256_file(PROBES)
    write_json(MANIFEST, man)
    print({k: man[k] for k in ("model_facts", "qual_prompts")})


if __name__ == "__main__":
    main()
