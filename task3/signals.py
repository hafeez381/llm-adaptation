"""C22-C23: rollout quality signals (equations 9-11, 13).

Token arrays are per rollout, in generation order, with termination tokens already excluded by the export.
VisConf's calibration pools the token KLs of the revealed candidates only, so it is recomputed per prefix.
"""
import numpy as np


def mean_logp(logp: np.ndarray) -> float:
    return float(np.mean(logp))


def self_certainty(kl_u_p: np.ndarray) -> float:
    return float(np.mean(kl_u_p))


def visual_engagement_tokens(a_img, a_prompt, a_gen) -> np.ndarray:
    a_img, a_prompt, a_gen = (np.asarray(x, dtype=float) for x in (a_img, a_prompt, a_gen))
    den = a_img + a_prompt + a_gen
    return np.divide(a_img, den, out=np.zeros_like(a_img), where=den > 0)


def calibrated_conf(kls: list[np.ndarray]) -> list[np.ndarray]:
    """c_{i,t} for every revealed rollout, with percentiles over the pooled tokens of exactly these rollouts."""
    pooled = np.concatenate(kls)
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    den = max(q95 - q05, 1e-8)
    return [np.clip((k - q05) / den, 0.0, 1.0) for k in kls]


def visconf(kls: list[np.ndarray], vis_tokens: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(Z_i = V_i C_i, Z_i^token = mean_t c v) for the revealed prefix."""
    cs = calibrated_conf(kls)
    z = np.array([c.mean() * v.mean() for c, v in zip(cs, vis_tokens)])
    zt = np.array([(c * v).mean() for c, v in zip(cs, vis_tokens)])
    return z, zt
