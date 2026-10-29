"""C21/C22 Verification on a synthetic pool: prefix-only decisions, nested coverage, N = 1 agreement, token costs."""
import numpy as np

from task3.offline import BUDGETS, prefix_fields, question_rows, select_all
from task3.selection import greedy_groups

EQ = lambda a, b: a == b


def fake_q(seed=0, vlm=True):
    rng = np.random.default_rng(seed)
    keys = [str(k) for k in rng.integers(0, 4, 32)]
    T = rng.integers(5, 40, 32).tolist()
    q = {"sid": f"q{seed}", "keys": keys, "T": T, "L": rng.normal(size=32).tolist(), "K": rng.random(32).tolist(),
         "correct": [k == "1" for k in keys], "kl": [rng.random(t).astype(np.float32) for t in T]}
    if vlm:
        q["v"] = [rng.random(t).astype(np.float32) for t in T]
        q["V"] = [float(v.mean()) for v in q["v"]]
    return q


def test_prefix_only_property():
    """assert_prefix_only: rewriting every rollout after position n never changes a decision at n."""
    q = fake_q()
    order = list(range(32))
    for n in BUDGETS[:-1]:
        g = greedy_groups([q["keys"][i] for i in order], EQ)
        a = select_all(prefix_fields(q, order, g, n, True), True)
        q2 = {k: (list(v) if isinstance(v, list) else v) for k, v in q.items()}
        rng = np.random.default_rng(99)
        for i in order[n:]:
            q2["keys"][i] = "9"
            q2["K"][i] = q2["L"][i] = 1e9
            q2["kl"][i] = rng.random(5).astype(np.float32) * 1e3
            q2["v"][i] = np.ones(5, np.float32)
            q2["V"][i] = 1.0
            q2["T"][i] = 1
        g2 = greedy_groups([q2["keys"][i] for i in order], EQ)
        assert select_all(prefix_fields(q2, order, g2, n, True), True) == a


def test_coverage_nested_and_n1_agreement_and_costs():
    q = fake_q(1)
    orders = [np.random.default_rng(s).permutation(32).tolist() for s in range(10)]
    rows = question_rows(q, orders, "heldout", EQ, True)
    by = {(r["N"], r["method"]): r for r in rows}
    cov = [by[(n, "sc")]["oracle"] for n in BUDGETS]
    assert all(a <= b for a, b in zip(cov, cov[1:]))
    n1 = {r["method"]: r["acc"] for r in rows if r["N"] == 1}
    assert len(set(n1.values())) == 1 and n1["first"] == by[(1, "sc")]["oracle"]
    assert by[(1, "first")]["cost"] == np.mean([q["T"][o[0]] for o in orders])
    assert by[(32, "sc")]["cost"] == sum(q["T"])
    for r in rows:  # sampling failures + selection failures + correct = all cases
        assert abs(r["fail_sampling"] + r["fail_selection"] + r["acc"] - 1) < 1e-12
