"""Where OPLoRA's residual overlap with the top-k subspace sits (follow-up to the C8 constraint check).

python -m task1.oplora_leak --store hub:<repo>
Training took U_k, V_k from a GPU FP32 SVD. The C8 check recomputes them with a CPU SVD. Singular values near
index k nearly tie, so the two top-k subspaces can differ by a small rotation at the boundary. This script uses a
float64 SVD as reference and reports, per layer: the singular-value gap at k, the factors' overlap with the
top-k subspace, the share of that overlap in the last 4 / 16 directions, and how much dW changes if A_eff is
re-projected in float64.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM

from common.hub_sync import make_store
from common.results import ROOT, write_json
from task1.diagnostics import K, LAYERS, NAME
from task1.merge import MODEL, revision


def leak_report(W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float, k: int = K) -> dict:
    W0, A, B = W0.double(), A.double(), B.double()
    U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
    Uk, Vk = U[:, :k], Vh[:k].T
    colA, colB = (A @ Vk).norm(dim=0) ** 2, (Uk.T @ B).norm(dim=1) ** 2
    dW = scale * B @ A
    A2 = A - (A @ Vk) @ Vk.T
    dW2 = scale * B @ A2
    share = lambda c, n: float(c[-n:].sum() / c.sum()) if c.sum() else 0.0
    return {"sigma_k_over_k1": float(S[k - 1] / S[k]),
            "top_k_energy_share": float((S[:k] ** 2).sum() / (S ** 2).sum()), "top_k_dim_share": k / W0.shape[0],
            "A_overlap": float(colA.sum().sqrt() / A.norm()), "B_overlap": float(colB.sum().sqrt() / B.norm()),
            "A_share_last4": share(colA, 4), "A_share_last16": share(colA, 16),
            "B_share_last4": share(colB, 4), "B_share_last16": share(colB, 16),
            "right_leak": float((dW @ Vk).norm() / dW.norm()),
            "right_leak_after_reprojection": float((dW2 @ Vk).norm() / dW2.norm()),
            "dW_rel_change_from_reprojection": float((dW2 - dW).norm() / dW.norm())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    store = make_store(ap.parse_args().store)
    path = ROOT / "outputs/diag/oplora16_adapters.pt"
    if not path.exists():
        store.get("task1/oplora16/final/adapters.pt", path)
    ex = torch.load(path, weights_only=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision(), dtype=torch.float32)
    out = {}
    for l in LAYERS:
        e = ex[NAME.format(l)]
        out[l] = leak_report(base.get_submodule(NAME.format(l)).weight.detach(), e["A_eff"], e["B_eff"], e["scale"])
    write_json(ROOT / "results/task1/diagnostics/oplora_leak.json", {"k": K, "reference": "float64 CPU SVD", "layers": out})


if __name__ == "__main__":
    main()
