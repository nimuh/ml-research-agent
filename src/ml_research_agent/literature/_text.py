"""Stdlib-only text utilities shared by query planning, dedup and ranking.

Deliberately dependency-free: this layer must never embed or call a model, so
every text signal it computes is lexical and reproducible.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-+][a-z0-9]+)*")

# Small, hand-kept list: an English stoplist plus the words every ML abstract
# contains, which carry no discriminative signal for retrieval or ranking.
_STOPWORD_TEXT = (
    "a an the and or but if then than that this these those there here it its is are was were "
    "be been being of in on at to for from by with without within into over under across "
    "about as via using use used uses we our us they their he she his her you your i not no "
    "nor do does did done can could may might must will would shall should have has had "
    "between among during per each other others such same both all any more most much many "
    "few less least very only also however therefore thus while when where which who whom "
    "what how why new novel approach approaches method methods model models paper study "
    "studies work works result results show shows shown propose proposed proposes present "
    "presents presented based towards toward framework system systems technique techniques "
    "analysis empirical experiments experimental"
)
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

_ACRONYM_STOP = frozenset({"e.g", "i.e", "et", "al"})


def strip_accents(text: str) -> str:
    """NFKD-fold accents so ``Schölkopf`` and ``Scholkopf`` compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str, *, drop_stopwords: bool = True, min_length: int = 2) -> list[str]:
    """Lowercase alphanumeric tokens, hyphen-joined compounds kept intact."""
    tokens = _TOKEN_RE.findall(strip_accents(text).lower())
    out = []
    for token in tokens:
        if len(token) < min_length and not token.isdigit():
            continue
        if drop_stopwords and (token in STOPWORDS or token in _ACRONYM_STOP):
            continue
        out.append(token)
    return out


def term_frequencies(tokens: Iterable[str]) -> dict[str, float]:
    """Sublinear term frequency (``1 + log tf``), which damps repeated jargon."""
    counts = Counter(tokens)
    return {term: 1.0 + math.log(count) for term, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity between two sparse weight maps; 0.0 if either is empty."""
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
    if dot == 0.0:
        return 0.0
    norm_left = math.sqrt(sum(w * w for w in left.values()))
    norm_right = math.sqrt(sum(w * w for w in right.values()))
    return dot / (norm_left * norm_right)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def key_phrase(text: str, *, max_terms: int = 6) -> str:
    """The content-bearing head of a sentence, used to turn a claim into a query."""
    tokens = tokenize(text)
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_terms:
            break
    return " ".join(seen)


__all__ = [
    "STOPWORDS",
    "collapse_whitespace",
    "cosine",
    "key_phrase",
    "strip_accents",
    "term_frequencies",
    "tokenize",
]
