import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9_.$%/-]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "is", "are", "was", "were", "be", "been", "being", "am", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "into", "that", "this", "these", "those",
    "it", "its", "i", "you", "we", "they", "he", "she", "them", "his", "her", "our",
    "your", "their", "do", "does", "did", "doing", "have", "has", "had", "having",
}

NEGATIONS = {"no", "not", "never", "none", "without", "cannot", "can't", "won't", "don't", "doesn't", "didn't"}
CONSTRAINT_WORDS = {
    "must", "should", "shall", "required", "require", "requires", "only", "unless", "except",
    "before", "after", "until", "avoid", "ensure", "mandatory", "prohibited", "allowed",
    "deny", "approve", "reject", "accept", "include", "exclude", "greater", "less", "minimum",
    "maximum", "atleast", "at", "least", "not", "never", "without", "cannot"
}


def words(text: str, keep_stopwords: bool = False) -> List[str]:
    toks = [w.lower() for w in WORD_RE.findall(text or "")]
    if keep_stopwords:
        return toks
    return [w for w in toks if w not in STOPWORDS]


def term_frequency_vector(text: str) -> Dict[str, float]:
    toks = words(text)
    c = Counter(toks)
    total = float(sum(c.values()) or 1.0)
    return {k: v / total for k, v in c.items()}


def binary_vector(text: str) -> Dict[str, float]:
    return {k: 1.0 for k in set(words(text))}


def cosine_from_dicts(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cosine_text(a: str, b: str) -> float:
    return cosine_from_dicts(term_frequency_vector(a), term_frequency_vector(b))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def shannon_entropy(text: str) -> float:
    toks = words(text, keep_stopwords=True)
    if not toks:
        return 0.0
    c = Counter(toks)
    total = sum(c.values())
    return -sum((v / total) * math.log2(v / total) for v in c.values())


def coverage(required: Sequence[str], candidate_text: str) -> float:
    req = [x.lower() for x in required if x]
    if not req:
        return 1.0
    cand = set(words(candidate_text, keep_stopwords=True))
    found = sum(1 for x in req if x.lower() in cand)
    return found / len(req)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def weighted_average(items: List[Tuple[float, float]]) -> float:
    denom = sum(w for _, w in items)
    if denom == 0:
        return 0.0
    return sum(v * w for v, w in items) / denom
