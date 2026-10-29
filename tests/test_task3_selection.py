"""C21/C22 Verification on hand-made candidates: RWC ranks and ties, prefix-only grouping and calibration."""
import numpy as np
import pytest

from task3.selection import (NO_ANSWER, greedy_groups, highest, rwc, rwc_margin, rwc_weights, sc_leader,
                             self_consistency)
from task3.signals import calibrated_conf, visconf


def test_rwc_toy_with_tied_scores():
    # 3 candidates, scores 0.9, 0.5, 0.5 -> descending ranks 1, 2.5, 2.5 -> w = (3-1)/2, (3-2.5)/2, (3-2.5)/2
    w = rwc_weights([0.9, 0.5, 0.5])
    assert np.allclose(w, [1.0, 0.25, 0.25])
    # groups A, B, B: W(A) = 1.0 > W(B) = 0.5, so A wins although B has more votes
    assert rwc([0, 1, 1], [0.9, 0.5, 0.5]) == 0
    assert self_consistency([0, 1, 1]) == 1
    assert rwc_margin([0, 1, 1], [0.9, 0.5, 0.5]) == pytest.approx((1.0 - 0.5) / 1.5)


def test_ties_go_to_earliest_supporting_rollout():
    assert self_consistency([1, 0, 0, 1]) == 0      # both groups have 2 votes; group 1 appeared first
    assert highest([0.3, 0.7, 0.7]) == 1
    assert rwc([0, 1], [0.2, 0.2]) == 0             # equal ranks 1.5 -> equal weights -> earliest


def test_single_candidate_every_rule_returns_it():
    assert self_consistency([0]) == highest([0.1]) == rwc([0], [0.1]) == 0
    assert rwc_margin([0], [0.1]) == 1.0


def test_no_answer_is_one_group_and_never_merges():
    eq = lambda a, b: a.strip("0") == b.strip("0")
    assert greedy_groups(["1", NO_ANSWER, "10", NO_ANSWER, "1"], eq) == [0, 1, 0, 1, 0]


def test_grouping_is_prefix_only():
    eq = lambda a, b: abs(float(a) - float(b)) < 0.15     # not transitive: 1.0 ~ 1.1 ~ 1.2
    order = ["1.0", "1.2", "1.1", "5"]
    full = greedy_groups(order, eq)
    for n in range(1, len(order) + 1):
        assert greedy_groups(order[:n], eq) == full[:n]
    # changing a later candidate never changes the earlier group ids
    assert greedy_groups(order[:2] + ["9", "9"], eq)[:2] == full[:2]


def test_sc_leader_uniqueness():
    assert sc_leader([0, 0, 1]) == (0, 2, True)
    assert sc_leader([0, 1]) == (0, 1, False)


def test_visconf_uses_only_revealed_tokens():
    rng = np.random.default_rng(0)
    kls = [rng.random(20) for _ in range(6)]
    vis = [rng.random(20) for _ in range(6)]
    z4, _ = visconf(kls[:4], vis[:4])
    kls_changed = kls[:4] + [k * 100 for k in kls[4:]]
    z4b, _ = visconf(kls_changed[:4], vis[:4])
    assert np.allclose(z4, z4b)
    assert all(((c >= 0) & (c <= 1)).all() for c in calibrated_conf(kls))


def test_tokenwise_fusion_covariance_identity():
    rng = np.random.default_rng(1)
    kls = [rng.random(30) for _ in range(5)]
    vis = [rng.random(30) for _ in range(5)]
    z, zt = visconf(kls, vis)
    cs = calibrated_conf(kls)
    cov = np.array([np.mean(c * v) - c.mean() * v.mean() for c, v in zip(cs, vis)])
    assert np.allclose(zt - z, cov, atol=1e-12)
