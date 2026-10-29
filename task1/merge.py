"""C7: resolve each evaluation condition to an FP16 model folder; merge adapters (OPLoRA: projected factors).

python -m task1.merge --condition {baseline,full_ft@625,full_ft,l2sp,lora4,lora16,oplora16} --store hub:<repo>
Adapters merge in FP32 (W0 upcast + scale * B_eff A_eff), then cast to FP16 (S4). The merged model's logits are
compared with the unmerged adapter model's on 8 probe prompts and the result is saved.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.hub_sync import make_store
from common.results import ROOT, read_json, write_json

MODEL = "Qwen/Qwen3-0.6B"
ADAPTERS = {"lora4", "lora16", "oplora16"}
CONDITIONS = ["baseline", "full_ft@625", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]


def revision() -> str:
    return read_json(ROOT / "data/manifest.json")["revisions"][f"hf_model:{MODEL}"]


def fetch_folder(store, prefix: str, local: Path, names) -> Path:
    for n in names:
        if not (local / n).exists():
            store.get(f"{prefix}/{n}", local / n)
    return local


def merge_adapters(model, exported: dict) -> None:
    """In-place W <- W0 + scale * B_eff @ A_eff for every adapted module, computed in FP32."""
    with torch.no_grad():
        for name, f in exported.items():
            w = model.get_submodule(name).weight
            w.copy_((w.float() + f["scale"] * (f["B_eff"].to(w.device) @ f["A_eff"].to(w.device))).to(w.dtype))


def probe_texts(tok, n=8) -> list[str]:
    from datasets import load_dataset
    ids = read_json(ROOT / "data/probe_ids.json")["gsm8k_test"][:n]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    return [tok.apply_chat_template([{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}],
                                    add_generation_prompt=True, enable_thinking=False, tokenize=False) for i in ids]


def merge_check(merged, unmerged, tok, device) -> dict:
    """S4: max |logit difference| and argmax agreement over every position of 8 probe prompts."""
    diffs, agree, total = [], 0, 0
    with torch.no_grad():
        for text in probe_texts(tok):
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            a = merged(input_ids=ids).logits.float()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                b = unmerged(input_ids=ids).logits.float()
            diffs.append((a - b).abs().max().item())
            agree += int((a.argmax(-1) == b.argmax(-1)).sum())
            total += ids.shape[1]
    out = {"max_abs_logit_diff": max(diffs), "argmax_agreement": agree / total, "positions": total}
    out["pass"] = out["max_abs_logit_diff"] <= 5e-2 and out["argmax_agreement"] >= 0.995
    return out


def resolve(condition: str, store, work: Path, device) -> Path:
    """Returns a local folder that lm-eval can load with dtype=float16."""
    out = work / condition.replace("@", "_")
    if (out / "config.json").exists() and any(out.glob("*.safetensors")):
        return out
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision())
    if condition == "baseline":
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision(), dtype=torch.float16)
    elif condition in ("full_ft@625", "full_ft", "l2sp"):
        run, tag = ("full_ft", "step-0625") if condition == "full_ft@625" else (condition, "final")
        src = fetch_folder(store, f"task1/{run}/{tag}", work / "src" / condition.replace("@", "_"),
                           ["config.json", "generation_config.json", "model.safetensors"])
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16)
    else:
        from task1.adapters import attach_adapters, cast_backbone_fp16
        from task1.train import load_cfg
        exported = torch.load(fetch_folder(store, f"task1/{condition}/final", work / "src" / condition,
                                           ["adapters.pt"]) / "adapters.pt", weights_only=False)
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision(), dtype=torch.float32).to(device)
        merge_adapters(m, exported)
        m = m.half()
        # unmerged reference: FP16 frozen backbone + FP32 adapters holding the exported effective factors
        cfg = load_cfg(condition, "full") | {"project": False}
        u = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision(), dtype=torch.float32).to(device)
        u.requires_grad_(False)
        attach_adapters(u, cfg)
        cast_backbone_fp16(u)
        from task1.adapters import lora_modules
        with torch.no_grad():
            for n, mod in lora_modules(u):
                mod.lora_A.copy_(exported[n]["A_eff"])
                mod.lora_B.copy_(exported[n]["B_eff"])
        check = merge_check(m.to(device).eval(), u.to(device).eval(), tok, device)
        write_json(ROOT / f"results/task1/eval/{condition}/merge_check.json", check)
        assert check["pass"], check
        del u
    m.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(resolve(args.condition, make_store(args.store), ROOT / "outputs/models", device))


if __name__ == "__main__":
    main()
