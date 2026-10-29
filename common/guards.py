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


def assert_backbone_frozen(model, trainable_names: set[str]) -> None:
    """Exactly `trainable_names` require grad, and no bias trains."""
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != trainable_names:
        extra, missing = sorted(actual - trainable_names), sorted(trainable_names - actual)
        raise GuardError(f"trainable set wrong: extra={extra[:3]} missing={missing[:3]}")
    biases = [n for n in actual if n.endswith(".bias")]
    if biases:
        raise GuardError(f"biases train: {biases[:3]}")
