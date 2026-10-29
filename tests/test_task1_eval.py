"""C7 checks that need no model: GSM8K per-filter sample merge, truncation flag, zero-shot guard, merge arithmetic."""
import pytest
import torch
from transformers import AutoTokenizer

from common.guards import GuardError
from task1.evaluate import assert_zero_shot, gsm8k_rows
from task1.merge import merge_adapters


def test_gsm8k_rows_merge_filters_and_flag_truncation():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    long = "word " * 40
    samples = []
    for d, resp in [(1, "So the answer is 18."), (0, long)]:
        for f, ext, em in [("strict-match", "[invalid]", 0.0), ("flexible-extract", "18", 1.0)]:
            samples.append({"doc_id": d, "arguments": [["P"]], "resps": [[resp]], "filtered_resps": [ext],
                            "filter": f, "exact_match": em, "target": "18"})
    rows = gsm8k_rows({"samples": {"gsm8k_cot": samples}}, tok, cap=30)
    assert [r["doc_id"] for r in rows] == [0, 1]
    assert rows[1]["exact_match_flexible_extract"] == 1.0 and rows[1]["exact_match_strict_match"] == 0.0
    assert rows[1]["extracted_flexible_extract"] == "18"
    assert rows[0]["truncated"] and not rows[1]["truncated"]


def test_zero_shot_guard():
    assert_zero_shot({"num_fewshot": {"gsm8k_cot": 0}, "gsm8k_chat_template": True})
    with pytest.raises(GuardError):
        assert_zero_shot({"num_fewshot": {"gsm8k_cot": 8}, "gsm8k_chat_template": True})


def test_merge_adds_scaled_effective_product():
    lin = torch.nn.Linear(6, 4, bias=False)
    model = torch.nn.Module()
    model.add_module("proj", lin)
    w0 = lin.weight.detach().clone()
    A, B = torch.randn(2, 6), torch.randn(4, 2)
    merge_adapters(model, {"proj": {"A_eff": A, "B_eff": B, "scale": 2.0}})
    assert torch.allclose(lin.weight, w0 + 2.0 * B @ A, atol=1e-6)
