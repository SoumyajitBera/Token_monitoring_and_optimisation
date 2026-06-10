import re
from typing import Tuple, Dict
from ..math_utils import NEGATIONS, CONSTRAINT_WORDS

SAFE_PHRASES = [
    "please", "kindly", "basically", "actually", "literally", "just", "really",
    "i would like to", "i want to", "can you", "could you", "would you", "as you know",
    "in order to", "at the end of the day", "to be honest", "you know", "sort of", "kind of",
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_normalize(text: str) -> Tuple[str, Dict[str, int]]:
    """Remove only safe filler. Does not delete negations or constraint-bearing words."""
    if not text:
        return "", {"removed_phrases": 0}
    out = text
    removed = 0
    for phrase in SAFE_PHRASES:
        low = phrase.lower()
        if low in NEGATIONS or low in CONSTRAINT_WORDS:
            continue
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        out, n = pattern.subn("", out)
        removed += n
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = normalize_whitespace(out)
    return out, {"removed_phrases": removed}
