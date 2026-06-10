from typing import Dict, List, Tuple
from ..core.schemas import Chunk
from ..math_utils import cosine_text
from ..tokenization import count_tokens


def filter_relevant_chunks(query: str, chunks: List[str], threshold: float = 0.15) -> Tuple[List[Chunk], List[Chunk], Dict[str, object]]:
    selected: List[Chunk] = []
    removed: List[Chunk] = []
    for i, text in enumerate(chunks or []):
        sim = cosine_text(query or "", text or "")
        ch = Chunk(text=text, metadata={"index": i, "similarity": sim}, value=sim, token_cost=count_tokens(text))
        if sim >= threshold or not query:
            selected.append(ch)
        else:
            removed.append(ch)
    selected.sort(key=lambda c: c.value, reverse=True)
    return selected, removed, {"threshold": threshold, "selected": len(selected), "removed": len(removed)}
