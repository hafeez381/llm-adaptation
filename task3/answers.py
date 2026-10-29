"""C20: answer normalization and equivalence, one function per dataset, used for grouping and for grading.

MATH: math-verify equivalence on the supplied extracted strings.
MathVista (D17): testmini.json metadata. multi_choice -> option letter (bare letter within the choice count, a
parenthesised letter, or text equal to a choice); integer -> int; float -> rounded to `precision`; text/list ->
normalized string. Anything that cannot be mapped stays its own string (own group, graded incorrect).
A missing or empty extraction is NO_ANSWER, always incorrect.
"""
import re
from functools import lru_cache

from task3.selection import NO_ANSWER

SCORER_VERSION = "task3-answers-v1"


def is_missing(a) -> bool:
    return a is None or not str(a).strip()


# ---------- MATH ----------
@lru_cache(maxsize=None)
def _parse_math(s: str):
    from math_verify import parse
    return parse(f"${s}$")


@lru_cache(maxsize=None)
def math_equiv(a: str, b: str) -> bool:
    if a == b:
        return True
    from math_verify import verify
    pa, pb = _parse_math(a), _parse_math(b)
    return bool(pa) and bool(pb) and verify(pa, pb)


def math_key(a) -> str:
    return NO_ANSWER if is_missing(a) else str(a).strip()


# ---------- MathVista ----------
_LATEX = [(r"\\text\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\\[,;:! ]", " "), (r"\\%", "%"),
          (r"\$", ""), (r"\\left|\\right", "")]
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def _clean(s: str) -> str:
    s = str(s)
    for pat, rep in _LATEX:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def _first_number(s: str):
    m = _NUM.search(_clean(s))
    return float(m.group().replace(",", "")) if m else None


def mathvista_key(a, meta: dict) -> str:
    """Canonical string for one answer under this question's metadata. Same function for gold."""
    if is_missing(a):
        return NO_ANSWER
    s = _clean(a)
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"] or []
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.fullmatch(r"\(?([A-Za-z])\)?\.?", s)
        if m and m.group(1).upper() in letters:
            return m.group(1).upper()
        for L, c in zip(letters, choices):
            if s.casefold() == _clean(c).casefold():
                return L
        return f"UNMATCHED:{s}"
    t = meta["answer_type"]
    if t in ("integer", "float"):
        x = _first_number(s)
        if x is None:
            return f"UNPARSED:{s}"
        if t == "integer":
            return str(int(x)) if x == int(x) else f"NONINT:{x}"
        p = int(meta.get("precision") or 0)
        return f"{round(x, p):.{p}f}"
    return re.sub(r"\s+", " ", s.casefold()).strip()


def mathvista_equiv(a: str, b: str) -> bool:
    return a == b  # keys are already canonical
