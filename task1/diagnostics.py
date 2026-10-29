"""C8: weight-space (rho_k, eta) and activation (R(z)) diagnostics on down_proj in layers 0, 13, 27.

python -m task1.diagnostics --store hub:<repo> [--mode smoke]
W0, U_k and the cached inputs z all come from the released checkpoint upcast to FP32 (S6). z is the down_proj
input at the last prompt token of each probe (D8), cached once from the starting model and reused for every model.
Full FT variants: dW = W_trained - W0 (FP32). Adapters: dW = scale * B_eff @ A_eff (projected for OPLoRA).
"""
import argparse
import hashlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.hub_sync import make_store
from common.results import ROOT, read_json, write_json
from task1.merge import MODEL, revision

LAYERS, K = [0, 13, 27], 128
MODELS = ["full_ft@625", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]
NAME = "model.layers.{}.mlp.down_proj"


def probe_prompts(tok) -> dict[str, list[str]]:
    """GSM8K: the GSM8K eval's non-thinking chat prompt. HellaSwag: the harness's plain context string."""
    from datasets import load_dataset
    from lm_eval.tasks.hellaswag.utils import process_docs
    p = read_json(ROOT / "data/probe_ids.json")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = process_docs(load_dataset("Rowan/hellaswag", split="validation").select(p["hellaswag_validation"]))
    math = [tok.apply_chat_template([{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}],
                                    add_generation_prompt=True, enable_thinking=False, tokenize=False)
            for i in p["gsm8k_test"]]
    return {"math": math, "nonmath": [d["query"] for d in hs]}


def cache_z(model, tok, prompts) -> dict[int, np.ndarray]:
    """down_proj inputs at the last prompt token; one prompt per forward, so there is no padding to skip."""
    store, hooks = {}, []
    for l in LAYERS:
        def hook(_m, inp, l=l):
            store.setdefault(l, []).append(inp[0][0, -1].detach().float().cpu().numpy())
        hooks.append(model.get_submodule(NAME.format(l)).register_forward_pre_hook(hook))
    with torch.no_grad():
        for text in prompts:
            model(input_ids=tok(text, return_tensors="pt").input_ids)
    for h in hooks:
        h.remove()
    return {l: np.stack(v) for l, v in store.items()}


def delta_full(store, run_tag: tuple[str, str], W0: dict) -> dict[int, np.ndarray]:
    from safetensors import safe_open
    run, tag = run_tag
    path = ROOT / f"outputs/diag/{run}_{tag}.safetensors"
    if not path.exists():
        store.get(f"task1/{run}/{tag}/model.safetensors", path)
    with safe_open(path, "pt") as f:
        return {l: f.get_tensor(NAME.format(l) + ".weight").float().numpy() - W0[l] for l in LAYERS}


def delta_adapter(store, run: str) -> dict[int, np.ndarray]:
    path = ROOT / f"outputs/diag/{run}_adapters.pt"
    if not path.exists():
        store.get(f"task1/{run}/final/adapters.pt", path)
    ex = torch.load(path, weights_only=False)
    return {l: (ex[NAME.format(l)]["scale"] * ex[NAME.format(l)]["B_eff"] @ ex[NAME.format(l)]["A_eff"]).numpy()
            for l in LAYERS}


def displacement(store, run_tag: tuple[str, str], base_sd: dict) -> float:
    """||theta - theta0||_2 over the unique parameters (the tied embedding counted once), read tensor by tensor."""
    from safetensors import safe_open
    run, tag = run_tag
    path = ROOT / f"outputs/diag/{run}_{tag}.safetensors"
    if not path.exists():
        store.get(f"task1/{run}/{tag}/model.safetensors", path)
    total = 0.0
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        assert keys <= set(base_sd), keys - set(base_sd)
        for k in keys:
            total += float((f.get_tensor(k).double() - base_sd[k].double()).pow(2).sum())
    return total ** 0.5


def metrics(dW, W0, Uk, Vk, z) -> dict:
    nrm2 = float((dW ** 2).sum())
    proj = Uk @ (Uk.T @ dW)
    R = np.linalg.norm(z @ dW.T, axis=1) / (np.linalg.norm(z @ W0.T, axis=1) + 1e-8)
    return {"rho_k": float((proj ** 2).sum() / nrm2) if nrm2 else 0.0,
            "eta": float(np.sqrt(nrm2) / np.linalg.norm(W0)),
            "right_leak": float(np.linalg.norm(dW @ Vk) / np.sqrt(nrm2)) if nrm2 else 0.0, "R": R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    store = make_store(args.store)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision())
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision(), dtype=torch.float32).eval()
    prompts = probe_prompts(tok)
    z = {k: cache_z(base, tok, v) for k, v in prompts.items()}
    for k in z:
        for l in LAYERS:
            assert not np.isnan(z[k][l]).any() and len({row.tobytes() for row in z[k][l]}) == len(z[k][l]), "D8 gate"
    z_hash = hashlib.sha256(b"".join(z[k][l].tobytes() for k in sorted(z) for l in LAYERS)).hexdigest()
    W0 = {l: base.get_submodule(NAME.format(l)).weight.detach().numpy().copy() for l in LAYERS}
    (ROOT / "outputs/diag").mkdir(parents=True, exist_ok=True)
    base_sd = {k: v.detach() for k, v in base.state_dict().items()}
    disp = {m: displacement(store, rt, base_sd) for m, rt in
            [("full_ft@625", ("full_ft", "step-0625")), ("full_ft", ("full_ft", "final")), ("l2sp", ("l2sp", "final"))]}
    write_json(ROOT / "results/task1/displacement.json", disp | {"l2sp_over_full_ft": disp["l2sp"] / disp["full_ft"]})
    del base_sd
    W0_fp16_err = {l: float(np.abs(W0[l].astype(np.float16).astype(np.float32) - W0[l]).max()) for l in LAYERS}
    bases = {}
    for l in LAYERS:
        U, _, Vh = np.linalg.svd(W0[l], full_matrices=False)
        bases[l] = (U[:, :K], Vh[:K].T)
    del base
    (ROOT / "outputs/diag").mkdir(parents=True, exist_ok=True)

    rows, oplora = [], {}
    for m in MODELS:
        dW = delta_full(store, ("full_ft", "step-0625") if m == "full_ft@625" else (m, "final"), W0) \
            if m in ("full_ft@625", "full_ft", "l2sp") else delta_adapter(store, m)
        for l in LAYERS:
            r = metrics(dW[l], W0[l], *bases[l], np.concatenate([z["math"][l], z["nonmath"][l]]))
            Rm, Rn = r["R"][:32], r["R"][32:]
            rows.append({"model": m, "layer": l, "rho_k": r["rho_k"], "eta": r["eta"],
                         "R_math_mean": float(Rm.mean()), "R_math_median": float(np.median(Rm)),
                         "R_nonmath_mean": float(Rn.mean()), "R_nonmath_median": float(np.median(Rn)),
                         "R_math_over_nonmath": float(Rm.mean() / Rn.mean()) if Rn.mean() else None})
            if m == "oplora16":
                oplora[l] = {"rho_k": r["rho_k"], "dW_Vk_rel": r["right_leak"],
                             "pass": r["rho_k"] <= 1e-6 and r["right_leak"] <= 1e-6}
    out = ROOT / "results/task1/diagnostics"
    write_json(out / "weight_activation.json", {"k": K, "layers": LAYERS, "z_sha256": z_hash,
                                                "W0_fp16_vs_fp32_max_abs": W0_fp16_err, "rows": rows})
    write_json(out / "oplora_constraints.json", oplora)  # constraint check, kept apart from benchmark evidence
    for r in rows:
        print(r)
    print("OPLoRA constraints", oplora)


if __name__ == "__main__":
    main()
