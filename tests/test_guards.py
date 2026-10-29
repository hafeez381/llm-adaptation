"""C0/C1 Verification on the committed files, plus each guard raising on a planted violation."""
import os
import time

import pytest

from common.guards import (GuardError, assert_no_gsm8k_overlap, assert_probe_frozen, assert_source_order,
                           normalize_text)
from common.results import ROOT, read_json, sha256_json, write_json

MAN = read_json(ROOT / "data/manifest.json")


def test_probe_file_is_frozen_and_complete():
    probes = read_json(ROOT / "data/probe_ids.json")
    assert len(set(probes["gsm8k_test"])) == 32 and len(set(probes["hellaswag_validation"])) == 32
    assert_probe_frozen(ROOT / "data/probe_ids.json", MAN)


def test_probe_guard_raises(tmp_path):
    p = tmp_path / "probe.json"
    write_json(p, {"a": 1})
    with pytest.raises(GuardError):
        assert_probe_frozen(p, {"probe_ids_sha256": "0" * 64})
    old = tmp_path / "eval.json"
    write_json(old, {})
    os.utime(old, (time.time() - 100, time.time() - 100))
    with pytest.raises(GuardError):
        assert_probe_frozen(p, {"probe_ids_sha256": __import__("common.results").results.sha256_file(p)}, [old])


def test_qual_prompts_shape():
    q = MAN["qual_prompts"]
    assert q["n_prompts"] == 24 and len(q["plain_order"]) == 8


def test_train_ids_match_manifest():
    ids = read_json(ROOT / "data/task1_train_ids.json")
    d = MAN["task1_data"]
    assert len(ids) == d["n"] == 40_000 and len(d["valid_tokens"]) == 40_000
    assert sum(d["valid_tokens"]) == d["valid_tokens_total"] and d["max_formatted_len"] <= 2048
    assert_source_order(ids, d, [d["max_formatted_len"]], 2048)


def test_source_order_guard_raises():
    ids = [0, 2, 1]
    with pytest.raises(GuardError):
        assert_source_order(ids, {"ordered_ids_sha256": sha256_json(ids)}, [10], 2048)
    with pytest.raises(GuardError):
        assert_source_order([0, 1], {"ordered_ids_sha256": sha256_json([0, 1])}, [2049], 2048)


def test_gsm8k_overlap_guard():
    norm = {normalize_text("Janet has  3 ducks.")}
    assert_no_gsm8k_overlap(["Janet has 4 ducks."], norm)
    with pytest.raises(GuardError):
        assert_no_gsm8k_overlap(["JANET has 3\nducks."], norm)
