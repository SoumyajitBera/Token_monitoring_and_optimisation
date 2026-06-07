from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    # Practical approximation for English + code-heavy prompts.
    return max(1, int(len(text) / 4))


def normalize_ws(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_sentences(text: str) -> list[str]:
    # Keeps bullets and lines as atomic chunks; safer than naive period-only splitting.
    chunks: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        if len(block) < 240:
            chunks.append(block)
        else:
            parts = re.split(r"(?<=[.!?])\s+", block)
            chunks.extend([p.strip() for p in parts if p.strip()])
    return chunks


def lexical_similarity(a: str, b: str) -> float:
    aw = set(re.findall(r"[a-zA-Z0-9_+#.-]+", a.lower()))
    bw = set(re.findall(r"[a-zA-Z0-9_+#.-]+", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", text)
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "into", "about", "would", "should", "could",
        "candidate", "project", "experience", "role", "work", "using", "used", "also", "have", "will",
    }
    return {w.lower() for w in words if w.lower() not in stop}


def top_terms(text: str, k: int = 40) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", text)]
    stop = {"the", "and", "for", "with", "this", "that", "from", "into", "about", "would", "should", "could", "candidate"}
    c = Counter(w for w in words if w not in stop)
    return [w for w, _ in c.most_common(k)]
