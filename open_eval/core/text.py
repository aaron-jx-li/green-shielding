import re

_PUNCT_RE = re.compile(r"[^a-z0-9\s/+\-]")


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match(a: str, b: str) -> bool:
    """
    Conservative string match (fast):
      - exact normalized
      - substring containment if both are reasonably long
    """
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True
    if len(a_n) >= 6 and len(b_n) >= 6:
        return (a_n in b_n) or (b_n in a_n)
    return False
