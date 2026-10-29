"""C5/C6: one adapter module for LoRA and OPLoRA (working doc D2).

forward(x) = base(x) + (alpha/r) * (dropout(x) @ A_eff.T) @ B_eff.T
  LoRA   (project=False): A_eff = A,                  B_eff = B
  OPLoRA (project=True):  A_eff = A - (A V_k) V_k^T,   B_eff = B - U_k (U_k^T B)
U_k, V_k are the top-k singular vectors of W0, from one FP32 SVD, never a d x d projection matrix.
A is drawn from one seeded generator in a fixed module order, so OPLoRA starts from LoRA-16's exact A and B.
"""
import math
import time

import torch
import torch.nn as nn

from common.seeding import SEED


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, gen: torch.Generator,
                 project: bool = False, k: int = 0):
        super().__init__()
        self.base, self.r, self.scale, self.project = base, r, alpha / r, project
        d_out, d_in = base.weight.shape
        A = torch.empty(r, d_in)
        bound = 1 / math.sqrt(d_in)  # kaiming_uniform_(a=sqrt(5)), PEFT's default for A
        A.uniform_(-bound, bound, generator=gen)
        self.lora_A = nn.Parameter(A)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.dropout = nn.Dropout(dropout)
        if project:
            U, _, Vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
            self.register_buffer("U_k", U[:, :k].contiguous(), persistent=False)
            self.register_buffer("V_k", Vh[:k].T.contiguous(), persistent=False)

    def effective(self) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.lora_A, self.lora_B
        if self.project:
            A = A - (A @ self.V_k) @ self.V_k.T
            B = B - self.U_k @ (self.U_k.T @ B)
        return A, B

    def delta_weight(self) -> torch.Tensor:
        A, B = self.effective()
        return self.scale * (B.float() @ A.float())

    def forward(self, x):
        A, B = self.effective()
        h = self.dropout(x).to(A.dtype) if not torch.is_autocast_enabled() else self.dropout(x)
        return self.base(x) + (self.scale * ((h @ A.T) @ B.T)).to(x.dtype)


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def attach_adapters(model, cfg: dict) -> None:
    """Wrap every target projection in every layer. Call on the FP32 model; the caller then casts the base to FP16."""
    gen = torch.Generator().manual_seed(SEED)
    t0 = time.time()
    targets = [(name, mod) for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in cfg["lora_targets"]]
    for name, mod in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1],
                LoRALinear(mod, cfg["r"], cfg["alpha"], cfg["dropout"], gen, cfg.get("project", False), cfg.get("k", 0)))
    model.adapter_overhead = {"n_adapted_modules": len(targets)}
    if cfg.get("project"):
        mods = lora_modules(model)
        model.adapter_overhead |= {
            "svd_seconds": time.time() - t0,
            "basis_bytes": sum((m.U_k.numel() + m.V_k.numel()) * 4 for _, m in mods)}

    def export_adapters():
        t = time.time()
        with torch.no_grad():
            out = {}
            for n, m in lora_modules(model):
                A, B = m.effective()
                out[n] = {"A_eff": A.detach().float().cpu(), "B_eff": B.detach().float().cpu(), "scale": m.scale}
        model.adapter_overhead["export_seconds"] = time.time() - t
        return out
    model.export_adapters = export_adapters


def cast_backbone_fp16(model) -> None:
    """Frozen backbone to FP16; adapter factors and SVD bases stay FP32."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.data = p.data.half()
