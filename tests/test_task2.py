"""C12/C13/C17 Verification on a tiny random LM: sequence layout, no-image path, gradients, losses, scoring."""
import numpy as np
import pytest
import torch

from common.guards import GuardError, assert_backbone_frozen
from task2.final_eval import assert_mismatch_map_fixed, mismatch_map
from task2.train import load_cfg, stage_lr
from task2.vlm import VLM, Connector, Encoder, answer_nll, collate, example_loss, load_lm, tokenizer, user_text
from task2.vqa_norm import normalize, soft_accuracy

REV = None


@pytest.fixture(scope="module")
def vlm():
    tok = tokenizer(REV)
    lm = load_lm(REV, "cpu", load_cfg("smoke")["tiny_lm"]).float()
    torch.manual_seed(0)
    return VLM(lm, Connector(768, lm.config.hidden_size), Encoder(tok)), tok


def test_visual_positions_and_no_image_path(vlm):
    model, tok = vlm
    enc = model.enc
    s_img, p_img = enc.example(user_text("presence", "Does this image contain a dog?"), "yes", True)
    s_no, p_no = enc.example(user_text("presence", "Does this image contain a dog?"), "yes", False)
    assert s_img.count(enc.pad) == 196 and s_img.count(enc.vs) == 1 and s_img.count(enc.ve) == 1
    first, last = s_img.index(enc.pad), len(s_img) - 1 - s_img[::-1].index(enc.pad)
    assert last - first == 195                                              # one contiguous block
    q = tok("Does this image", add_special_tokens=False).input_ids
    assert s_img.index(q[0], last) > last                                     # question after the visual block
    user = tok.convert_tokens_to_ids("<|im_start|>")
    assert s_img.index(user) < first                                         # inside the user turn
    assert enc.pad not in s_no and enc.vs not in s_no and enc.ve not in s_no
    assert len(s_img) - len(s_no) == 196 + 3                                 # markers, pads and the newline
    assert s_img[p_img:] == s_no[p_no:]                                      # same target tokens
    # a complementary swap changes features only, never token ids
    s_swap, _ = enc.example(user_text("presence", "Does this image contain a dog?"), "yes", True)
    assert s_swap == s_img


def test_connector_gets_gradient_qwen_does_not(vlm):
    model, tok = vlm
    model.lm.requires_grad_(False)
    names = {f"connector.{n}" for n, _ in model.connector.named_parameters()}
    assert_backbone_frozen(model, names, bias_ok=("connector.",))
    seqs = [model.enc.example(user_text("caption", None), "a dog on grass", True),
            model.enc.example(user_text("presence", "Is it?"), "no", False)]
    ids, attn, labels = collate(seqs, tok.pad_token_id, "cpu")
    nll, keep = model.token_nll(ids, attn, labels, [torch.randn(196, 768), None])
    example_loss(nll, keep).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.connector.parameters())
    assert all(p.grad is None for p in model.lm.parameters())


def test_equation6_example_weighted_and_answer_nll_drops_eot():
    nll = torch.tensor([[0.0, 1.0, 3.0, 0.0], [2.0, 2.0, 2.0, 6.0]])
    keep = torch.tensor([[False, True, True, False], [True, True, True, True]])
    assert torch.allclose(example_loss(nll, keep), torch.tensor([2.0, 3.0]))   # per-example means, not token mean
    assert torch.allclose(answer_nll(nll, keep), torch.tensor([1.0, 2.0]))     # last supervised token removed


def test_soft_accuracy_hand_examples():
    answers = ["2"] * 3 + ["3"] * 7
    assert soft_accuracy("two", answers) == pytest.approx(0.9)               # 3 of 10 agree -> 0.9
    assert soft_accuracy("5", answers) == 0.0
    assert soft_accuracy("3", answers) == 1.0
    assert normalize("A Dog!") == "dog" and normalize("Two") == "2"


def test_stage_schedule_warmup_and_floor():
    cfg = load_cfg("full")
    n = 2250
    lrs = [stage_lr(cfg, 1e-3, j, n) for j in range(1, n + 1)]
    assert lrs[112] == pytest.approx(1e-3)                                   # ceil(5% of 2250) = 113 warmup updates
    assert lrs[-1] == pytest.approx(1e-4)                                    # cosine ends at 10% of peak


def test_mismatch_map_excludes_own_pair(tmp_path):
    pairs = [{"pair_id": f"p{i}", "items": [{"image_id": 2 * i}, {"image_id": 2 * i + 1}]} for i in range(5)]
    mm = mismatch_map(pairs, np.random.default_rng(0))
    path = tmp_path / "mm.json"
    path.write_text(__import__("json").dumps(mm))
    assert_mismatch_map_fixed(path, pairs)
    mm["p0|0"] = 1
    path.write_text(__import__("json").dumps(mm))
    with pytest.raises(GuardError):
        assert_mismatch_map_fixed(path, pairs)


def test_committed_splits_are_image_disjoint_and_sized():
    from common.guards import assert_split_disjoint
    from common.results import ROOT, read_json
    d = read_json(ROOT / "data/task2_ids.json")
    assert_split_disjoint(d["image_sets"])
    r = d["report"]
    assert r["sizes"] | {} == {"caption_train": 18000, "caption_dev": 96, "vqa_train": 1536, "vqa_dev": 32,
                              "vqa_test": 96, "presence_pool": 4000, "counting_pool": 4000, "grounding_dev": 24,
                              "grounding_test": 64}
    assert r["vqa_types"]["vqa_test"] == {"yes/no": 36, "number": 24, "other": 36}
    assert r["grounding_types"]["grounding_test"] == {"presence": 32, "counting": 32}
    with pytest.raises(GuardError):
        assert_split_disjoint({"caption_train": [1], "vqa_dev": [1]})


def test_left_padded_generation_matches_single(vlm):
    """Batched greedy generation from left-padded inputs_embeds equals unpadded generation per example."""
    from types import SimpleNamespace
    from task2.final_eval import generate
    model, tok = vlm
    g = torch.Generator().manual_seed(3)
    feats = {"1": torch.randn(196, 768, generator=g), "2": torch.randn(196, 768, generator=g)}
    tr = SimpleNamespace(model=model, tok=tok, feats=feats, device=torch.device("cpu"), amp=False)
    exs = [{"kind": "vqa", "question": "What?", "image_id": 1},
           {"kind": "vqa", "question": "What colour is the large object near the left edge of the picture?", "image_id": 2}]
    both = generate(tr, exs, lambda x: x["image_id"], True)
    alone = [generate(tr, [x], lambda x: x["image_id"], True)[0] for x in exs]
    assert [b["pred"] for b in both] == [a["pred"] for a in alone]


def test_task2_resume_matches_straight(tmp_path):
    """C13 gate: 4 updates + snapshot + restart + rest equals a straight run, with the same sampler draws."""
    from task2.train import Trainer, load_cfg, load_inputs
    cfg = load_cfg("smoke") | {"dev_eval_at": [], "gen_check_at": []}
    items, feats = load_inputs("smoke")
    a = Trainer(cfg, items, feats, str(tmp_path / "hub_a"), tmp_path / "a", "cpu")
    a.calibrate()
    a.fit()
    b = Trainer(cfg, items, feats, str(tmp_path / "hub_b"), tmp_path / "b", "cpu")
    b.calibrate()
    b.fit(stop_after=4)
    b.snapshot()
    c = Trainer(cfg, items, feats, str(tmp_path / "hub_b"), tmp_path / "c", "cpu")
    c.resume()
    c.fit()
    la = [r["loss"] for r in __import__("common.results").results.read_jsonl(tmp_path / "a/train_log.jsonl")]
    lc = [r["loss"] for r in __import__("common.results").results.read_jsonl(tmp_path / "c/train_log.jsonl")]
    ma = [r["mixture"] for r in __import__("common.results").results.read_jsonl(tmp_path / "a/train_log.jsonl")]
    mc = [r["mixture"] for r in __import__("common.results").results.read_jsonl(tmp_path / "c/train_log.jsonl")]
    assert len(la) == len(lc) == 10 and ma == mc
    assert max(abs(x - y) for x, y in zip(la, lc)) < 1e-3
