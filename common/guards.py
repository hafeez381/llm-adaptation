"""Leakage and integrity guards (working doc §2). Each one raises; none of them warns."""
import re
import unicodedata
from pathlib import Path

from common.results import sha256_file, sha256_json


class GuardError(AssertionError):
    pass


def normalize_text(s: str) -> str:
    """S1 reading: NFKC, casefold, collapse whitespace, strip. Punctuation is kept."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def assert_no_gsm8k_overlap(questions: list[str], gsm8k_test_norm: set[str]) -> None:
    hits = [q for q in questions if normalize_text(q) in gsm8k_test_norm]
    if hits:
        raise GuardError(f"{len(hits)} selected questions match GSM8K test, e.g. {hits[0][:80]!r}")


def assert_source_order(ids: list[int], manifest: dict, lengths: list[int], max_len: int) -> None:
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise GuardError("training IDs are not strictly increasing source indices")
    if sha256_json(ids) != manifest["ordered_ids_sha256"]:
        raise GuardError("ordered-ID sha256 differs from the manifest")
    if max(lengths) > max_len:
        raise GuardError(f"an example has {max(lengths)} tokens > {max_len}")


def assert_probe_frozen(probe_path, manifest: dict, eval_outputs: list = ()) -> None:
    probe_path = Path(probe_path)
    if sha256_file(probe_path) != manifest["probe_ids_sha256"]:
        raise GuardError("probe file sha256 differs from the manifest")
    born = probe_path.stat().st_mtime
    older = [str(p) for p in eval_outputs if Path(p).stat().st_mtime < born]
    if older:
        raise GuardError(f"eval outputs predate the probe file: {older[:3]}")


def assert_split_disjoint(image_sets: dict[str, list]) -> None:
    """Training (all pools together) and each dev / test set share no image with one another."""
    train = set().union(*(set(v) for k, v in image_sets.items() if k.endswith(("_train", "_pool"))))
    evals = {k: set(v) for k, v in image_sets.items() if k.endswith(("_dev", "_test"))}
    groups = {"training": train} | evals
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] & groups[b]:
                raise GuardError(f"{a} and {b} share {len(groups[a] & groups[b])} images")


def assert_oplora_init_equal(oplora_init: dict, lora16_init: dict, model, tol: float = 1e-5) -> None:
    """OPLoRA starts from LoRA-16's exact unprojected A and B; its bases are orthonormal."""
    import torch
    if oplora_init.keys() != lora16_init.keys():
        raise GuardError("OPLoRA and LoRA-16 adapt different modules")
    for n in oplora_init:
        if not all(torch.equal(oplora_init[n][f], lora16_init[n][f]) for f in ("A", "B")):
            raise GuardError(f"initial A/B differ at {n}")
    for n, m in model.named_modules():
        if getattr(m, "project", False):
            for basis in (m.U_k, m.V_k):
                eye = torch.eye(basis.shape[1], device=basis.device)
                err = (basis.T @ basis - eye).abs().max().item()
                if err > tol:
                    raise GuardError(f"basis not orthonormal at {n}: {err:.2e}")


def assert_backbone_frozen(model, trainable_names: set[str], bias_ok: tuple[str, ...] = ()) -> None:
    """Exactly `trainable_names` require grad, and no bias trains outside the `bias_ok` prefixes (the connector)."""
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != trainable_names:
        extra, missing = sorted(actual - trainable_names), sorted(trainable_names - actual)
        raise GuardError(f"trainable set wrong: extra={extra[:3]} missing={missing[:3]}")
    biases = [n for n in actual if n.endswith(".bias") and not n.startswith(bias_ok)]
    if biases:
        raise GuardError(f"biases train: {biases[:3]}")
