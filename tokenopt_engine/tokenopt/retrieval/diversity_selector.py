from typing import Dict, List, Tuple
from ..core.schemas import Chunk
from ..math_utils import cosine_text


def greedy_diversity_select(chunks: List[Chunk], max_similarity: float = 0.92) -> Tuple[List[Chunk], List[Chunk], Dict[str, object]]:
    selected: List[Chunk] = []
    removed: List[Chunk] = []
    for ch in sorted(chunks, key=lambda c: c.value, reverse=True):
        if any(cosine_text(ch.text, s.text) >= max_similarity for s in selected):
            removed.append(ch)
        else:
            selected.append(ch)
    return selected, removed, {"max_similarity": max_similarity, "selected": len(selected), "removed": len(removed)}
