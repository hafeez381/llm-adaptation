"""C2 Verification: loss normalization, accumulation, resume, parameter groups, schedule; C4's L2-SP gradient."""
import math

import pytest
import torch
import torch.nn.functional as F

from task1.train import Trainer, collate, load_cfg, lr_lambda, sft_loss_sum

VOCAB = 512


def tiny_cfg(method="full_ft", **over):
    cfg = load_cfg(method, "smoke")
    cfg["tiny_model"] = cfg["tiny_model"] | {"vocab_size": VOCAB}
    return cfg | over


def fake_data(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids, plen = [], []
    for _ in range(n):
        L = int(torch.randint(12, 40, (1,), generator=g))
        p = int(torch.randint(3, L - 2, (1,), generator=g))
        ids.append(torch.randint(1, VOCAB, (L,), generator=g).tolist())
        plen.append(p)
    return {"input_ids": ids, "prompt_len": plen, "pad_id": 0}


def make(tmp_path, name="run", **over):
    return Trainer(tiny_cfg(**over), fake_data(), str(tmp_path / "hub"), tmp_path / name, device="cpu", amp=False)


def test_loss_matches_hand_and_full_logits(tmp_path):
    """D1/D3 gate: gathered-lm_head loss = full-logits reference = hand-computed token mean."""
    tr = make(tmp_path)
    batch = [(tr.data["input_ids"][i], tr.data["prompt_len"][i]) for i in range(2)]
    ids, attn, labels = collate(batch, 0, "cpu")
    with torch.no_grad():
        s, n = sft_loss_sum(tr.model, ids, attn, labels)
        logits = tr.model(input_ids=ids, attention_mask=attn).logits.float()
        ref = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1),
                              ignore_index=-100, reduction="sum")
        hand, cnt = 0.0, 0
        for b, (seq, p) in enumerate(batch):
            lp = logits[b].log_softmax(-1)
            for t in range(p, len(seq)):  # token t is predicted from position t - 1
                hand -= float(lp[t - 1, seq[t]])
                cnt += 1
    assert n == cnt == sum(len(q) - p for q, p in batch)
    assert abs(float(s) / n - float(ref) / n) < 1e-4
    assert abs(float(s) / n - hand / cnt) < 1e-4


def test_accumulation_equals_one_pass(tmp_path):
    """16 x 2 with a shared denominator gives the 32-example gradient (here 8 x 2 vs 1 x 16)."""
    tr = make(tmp_path)
    batch = [(tr.data["input_ids"][i], tr.data["prompt_len"][i]) for i in range(16)]
    tr.accumulate(batch, microbatch=2)
    g_acc = [p.grad.clone() for p in tr.params]
    tr.opt.zero_grad(set_to_none=True)
    tr.accumulate(batch, microbatch=16)
    g_one = [p.grad for p in tr.params]
    num = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(g_acc, g_one)))
    den = math.sqrt(sum(float(b.pow(2).sum()) for b in g_one))
    assert num / den < 1e-3


def test_resume_matches_straight_run(tmp_path):
    """D6 gate: 8 straight updates vs 4 + snapshot + restart + 4; per-update losses match to 1e-3."""
    over = {"updates": 8, "grad_accum": 2, "warmup_updates": 2, "ckpt_updates": [4], "keep_weights_at": []}
    straight = [r["sft_loss"] for r in make(tmp_path, "a", **over).fit()]
    over_b = over | {"ckpt_updates": [4]}
    first = Trainer(tiny_cfg(**over_b), fake_data(), str(tmp_path / "hub_b"), tmp_path / "b", "cpu", amp=False)
    part1 = [r["sft_loss"] for r in first.fit(stop_after=4)]
    second = Trainer(tiny_cfg(**over_b), fake_data(), str(tmp_path / "hub_b"), tmp_path / "b2", "cpu", amp=False)
    second.resume()
    assert second.update == 4
    part2 = [r["sft_loss"] for r in second.fit()]
    assert len(part1 + part2) == 8
    assert max(abs(a - b) for a, b in zip(straight, part1 + part2)) < 1e-3


def test_no_bias_or_norm_in_decay_group(tmp_path):
    tr = make(tmp_path)
    decay, no_decay = tr.opt.param_groups
    assert all(p.ndim >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.01
    assert all(p.ndim < 2 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0
    print("decayed tensors", len(decay["params"]), "undecayed", len(no_decay["params"]))


def test_schedule_warmup_then_cosine_to_floor():
    f = lr_lambda({"warmup_updates": 38, "updates": 1250, "lr_floor": 0.0})
    lrs = [f(s) for s in range(1250)]  # lrs[u - 1] is the multiplier used at update u
    assert lrs[0] == pytest.approx(1 / 38) and lrs[37] == lrs[38] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[38:], lrs[39:]))
    assert 0 < lrs[-1] < 1e-5


def test_l2sp_gradient_equals_autograd():
    """C4 check: autograd of lam * Omega equals lam * (theta - theta0) to 1e-6."""
    torch.manual_seed(0)
    p0 = [torch.randn(5, 3), torch.randn(3)]
    p = [(x + 0.1 * torch.randn_like(x)).requires_grad_() for x in p0]
    lam = 1e-3
    omega = 0.5 * sum(((a - b) ** 2).sum() for a, b in zip(p, p0))
    (lam * omega).backward()
    for a, b in zip(p, p0):
        assert torch.allclose(a.grad, lam * (a.detach() - b), atol=1e-6)


def test_l2sp_omega_zero_before_first_step(tmp_path):
    tr = make(tmp_path, method="l2sp", lam=1e-3)
    row = tr.step()
    assert row["omega"] == 0.0 and row["total_loss"] == row["sft_loss"]
