"""C5/C6 Verification: parameter counts, PEFT equivalence, step-0 identity, OPLoRA constraints and init equality."""
import math
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common.guards import assert_backbone_frozen, assert_oplora_init_equal
from common.results import ROOT, read_json
from task1.adapters import attach_adapters, lora_modules
from task1.train import Trainer, build_model, load_cfg
from tests.test_task1_train import VOCAB, fake_data, tiny_cfg

FACTS = read_json(ROOT / "data/manifest.json")["model_facts"]


def one_real_layer():
    """Qwen3-0.6B's real projection shapes, 1 layer, small vocab: per-layer counts scale by 28."""
    mc = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    mc.num_hidden_layers, mc.vocab_size = 1, VOCAB
    mc.layer_types = mc.layer_types[:1]
    return AutoModelForCausalLM.from_config(mc, dtype=torch.float32)


def test_lora_param_counts_match_layer_shapes():
    h, i = FACTS["hidden_size"], FACTS["intermediate_size"]
    q = FACTS["num_attention_heads"] * FACTS["head_dim"]
    kv = FACTS["num_key_value_heads"] * FACTS["head_dim"]
    per_r = (h + q) + 2 * (h + kv) + (q + h) + 2 * (h + i) + (i + h)  # q, k+v, o, gate+up, down
    counts = {}
    for r in (4, 16):
        m = one_real_layer().requires_grad_(False)
        attach_adapters(m, tiny_cfg("lora16") | {"r": r, "alpha": 2 * r})
        counts[r] = FACTS["num_hidden_layers"] * sum(p.numel() for n, p in m.named_parameters() if "lora_" in n)
        assert counts[r] == FACTS["num_hidden_layers"] * per_r * r
    assert counts == {4: 2_523_136, 16: 10_092_544} and counts[16] == 4 * counts[4]


def test_matches_peft_lora():
    """D2 gate: project=False equals PEFT LoRA for the same A, B, alpha, r (dropout off)."""
    from peft import LoraConfig, get_peft_model
    cfg = tiny_cfg("lora16") | {"device": "cpu"}
    base = build_model(cfg | {"mode": "smoke", "kind": "full"}, "cpu").eval()
    for p in base.parameters():  # same FP16-rounded backbone as ours, so only the adapter math is compared
        p.data = p.data.half().float()
    ours = build_model(cfg, "cpu").float().eval()
    peft = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=cfg["lora_targets"]))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, m in lora_modules(ours):
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.02)
            pm = peft.get_submodule("base_model.model." + n)
            pm.lora_A["default"].weight.copy_(m.lora_A)
            pm.lora_B["default"].weight.copy_(m.lora_B)
        ids = torch.randint(1, VOCAB, (2, 12))
        assert (ours(input_ids=ids).logits - peft(input_ids=ids).logits).abs().max() < 1e-5


def test_step0_logits_equal_base_for_lora_and_oplora():
    ids = torch.randint(1, VOCAB, (2, 12))
    base = build_model(tiny_cfg("lora16") | {"kind": "full"}, "cpu").eval()
    with torch.no_grad():
        ref = base(input_ids=ids).logits
        for method in ("lora16", "oplora16"):
            m = build_model(tiny_cfg(method), "cpu").float().eval()
            assert torch.equal(m(input_ids=ids).logits, ref) or (m(input_ids=ids).logits - ref).abs().max() < 1e-2


def test_backbone_frozen_and_no_bias():
    m = build_model(tiny_cfg("lora16"), "cpu")
    assert_backbone_frozen(m, {n for n, _ in m.named_parameters() if ".lora_" in n})
    assert all(p.dtype == torch.float32 for n, p in m.named_parameters() if p.requires_grad)
    assert all(p.dtype == torch.float16 for n, p in m.named_parameters() if not p.requires_grad)


def test_oplora_init_equals_lora16_and_bases_orthonormal():
    lora = build_model(tiny_cfg("lora16"), "cpu")
    op = build_model(tiny_cfg("oplora16"), "cpu")
    init = lambda mdl: {n: {"A": m.lora_A.detach(), "B": m.lora_B.detach()} for n, m in lora_modules(mdl)}
    assert_oplora_init_equal(init(op), init(lora), op)


def test_oplora_constraints_after_20_updates(tmp_path):
    """D2 gate: after 20 updates the exported factors give rho_k <= 1e-6 and ||dW V_k|| / ||dW|| <= 1e-6."""
    over = {"updates": 20, "grad_accum": 2, "warmup_updates": 2, "ckpt_updates": [], "keep_weights_at": [], "lr": 5e-3}
    tr = Trainer(tiny_cfg("oplora16", **over), fake_data(n=80), str(tmp_path / "hub"), tmp_path / "op", "cpu", amp=False)
    tr.fit()
    exported = tr.model.export_adapters()
    w0_model = build_model(tiny_cfg("oplora16") | {"kind": "full"}, "cpu")  # FP32 W0, same seed (S6 reference)
    for n, m in lora_modules(tr.model):
        f = exported[n]
        dW = f["scale"] * f["B_eff"] @ f["A_eff"]
        W0 = w0_model.get_submodule(n).weight.detach()
        U, _, Vh = torch.linalg.svd(W0, full_matrices=False)  # bases recomputed independently of the adapter
        Uk, Vk = U[:, :8], Vh[:8].T
        nrm2 = dW.pow(2).sum()
        assert nrm2 > 0
        assert (Uk @ (Uk.T @ dW)).pow(2).sum() / nrm2 <= 1e-6
        assert (dW @ Vk).norm() / dW.norm() <= 1e-6


def test_oplora_leak_report_locates_boundary_rotation():
    """Factors projected against a basis rotated between directions k and k+1 leak only at the boundary."""
    from task1.oplora_leak import leak_report
    g = torch.Generator().manual_seed(0)
    k, d = 8, 32
    W0 = torch.randn(d, d, generator=g, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(W0)
    V = Vh.T.clone()
    c, s = math.cos(1e-3), math.sin(1e-3)
    V[:, [k - 1, k]] = V[:, [k - 1, k]] @ torch.tensor([[c, -s], [s, c]], dtype=torch.float64)  # "other SVD"
    A = torch.randn(4, d, generator=g, dtype=torch.float64)
    A = A - (A @ V[:, :k]) @ V[:, :k].T
    B = torch.randn(d, 4, generator=g, dtype=torch.float64)
    B = B - U[:, :k] @ (U[:, :k].T @ B)
    r = leak_report(W0, A, B, 1.0, k)
    assert 0 < r["A_overlap"] < 1e-2 and r["A_share_last4"] > 0.99
    assert r["B_overlap"] < 1e-12
    assert r["right_leak_after_reprojection"] < 1e-12
