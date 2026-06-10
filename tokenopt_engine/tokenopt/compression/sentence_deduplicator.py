from typing import Dict, List, Tuple
from ..math_utils import cosine_text
from ..tokenization import split_sentences


def deduplicate_sentences(text: str, threshold: float = 0.90) -> Tuple[str, Dict[str, object]]:
    sentences = split_sentences(text)
    kept: List[str] = []
    removed: List[str] = []
    for s in sentences:
        duplicate = False
        for k in kept:
            if cosine_text(s, k) >= threshold:
                duplicate = True
                break
        if duplicate:
            removed.append(s)
        else:
            kept.append(s)
    return "\n".join(kept), {"kept_sentences": len(kept), "removed_sentences": len(removed), "removed": removed}
