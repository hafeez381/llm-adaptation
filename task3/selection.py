"""C20-C22: answer groups and selection rules over a revealed prefix (working doc D16).

Every function here takes candidate fields only (group ids, scores, token counts in reveal order), never gold
or correctness. Ties go to the earliest supporting rollout in the current order.
"""
import numpy as np
from scipy.stats import rankdata

NO_ANSWER = "NO_ANSWER"


def greedy_groups(answers: list[str], equiv) -> list[int]:
    """Group ids in reveal order: each answer joins the first earlier group whose first member is equivalent.

    A group only ever depends on earlier answers, so the grouping of a prefix is the prefix of the grouping.
    """
    reps, gid = [], []
    for a in answers:
        for g, r in enumerate(reps):
            if (a == NO_ANSWER) == (r == NO_ANSWER) and (a == r or (a != NO_ANSWER and equiv(a, r))):
                gid.append(g)
                break
        else:
            reps.append(a)
            gid.append(len(reps) - 1)
    return gid


def _vote_winner(groups: np.ndarray, weights: np.ndarray) -> tuple[int, np.ndarray]:
    """Group with the largest summed weight; ties go to the group whose first member came earliest."""
    ids = np.unique(groups)
    totals = np.array([weights[groups == g].sum() for g in ids])
    first = np.array([np.argmax(groups == g) for g in ids])
    best = np.flatnonzero(np.isclose(totals, totals.max(), rtol=0, atol=1e-12))
    return int(ids[best[np.argmin(first[best])]]), totals


def self_consistency(groups) -> int:
    """Returns the position (in reveal order) of the first member of the winning group."""
    g = np.asarray(groups)
    win, _ = _vote_winner(g, np.ones(len(g)))
    return int(np.argmax(g == win))


def highest(scores) -> int:
    s = np.asarray(scores, dtype=float)
    return int(np.flatnonzero(s == s.max())[0])


def rwc_weights(scores) -> np.ndarray:
    """w_i = (N - r_i) / (N - 1) with descending average ranks (rank 1 = best score)."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 1:
        return np.ones(1)
    r = rankdata(-s, method="average")
    return (n - r) / (n - 1)


def rwc(groups, scores) -> int:
    g = np.asarray(groups)
    if len(g) == 1:
        return 0
    win, _ = _vote_winner(g, rwc_weights(scores))
    return int(np.argmax(g == win))


def rwc_margin(groups, scores) -> float:
    """(largest - second-largest answer vote) / total vote weight; runner-up 0 when one group exists."""
    g = np.asarray(groups)
    w = rwc_weights(scores)
    _, totals = _vote_winner(g, w)
    top = np.sort(totals)[::-1]
    second = top[1] if len(top) > 1 else 0.0
    return float((top[0] - second) / w.sum()) if w.sum() > 0 else 0.0


def sc_leader(groups) -> tuple[int, int, bool]:
    """(winning position, its support count, whether the lead is unique)."""
    g = np.asarray(groups)
    ids, counts = np.unique(g, return_counts=True)
    pos = self_consistency(g)
    return pos, int(counts.max()), int((counts == counts.max()).sum()) == 1
