import re
from typing import List

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Approximate token counter with no external dependency.

    This is intentionally conservative. For exact model counts you can later
    plug in tiktoken or provider-specific tokenizers.
    """
    if not text:
        return 0
    pieces = simple_tokenize(text)
    extra = 0
    for p in pieces:
        if len(p) > 12:
            extra += len(p) // 8
    return len(pieces) + extra


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]
