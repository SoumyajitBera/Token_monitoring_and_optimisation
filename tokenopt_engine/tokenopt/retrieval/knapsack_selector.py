from typing import Dict, List, Tuple
from ..core.schemas import Chunk


def select_chunks_knapsack(chunks: List[Chunk], budget_tokens: int) -> Tuple[List[Chunk], List[Chunk], Dict[str, object]]:
    """0/1 knapsack using value and token_cost. From scratch, no dependencies.

    To keep memory safe, uses a 1D DP array storing selected indexes.
    """
    if budget_tokens <= 0 or not chunks:
        return [], chunks, {"budget_tokens": budget_tokens, "selected": 0, "removed": len(chunks)}

    # Scale values to integers for deterministic DP.
    values = [max(0, int(round(c.value * 10000))) for c in chunks]
    weights = [max(1, c.token_cost) for c in chunks]

    dp = [0] * (budget_tokens + 1)
    picks = [set() for _ in range(budget_tokens + 1)]

    for idx, (w, v) in enumerate(zip(weights, values)):
        if w > budget_tokens:
            continue
        for cap in range(budget_tokens, w - 1, -1):
            cand = dp[cap - w] + v
            if cand > dp[cap]:
                dp[cap] = cand
                picks[cap] = set(picks[cap - w])
                picks[cap].add(idx)

    best_cap = max(range(budget_tokens + 1), key=lambda c: dp[c])
    selected_idx = picks[best_cap]
    selected = [chunks[i] for i in sorted(selected_idx, key=lambda i: chunks[i].metadata.get("index", i))]
    removed = [c for i, c in enumerate(chunks) if i not in selected_idx]
    return selected, removed, {
        "budget_tokens": budget_tokens,
        "used_tokens": sum(c.token_cost for c in selected),
        "selected": len(selected),
        "removed": len(removed),
        "objective_value": dp[best_cap] / 10000.0,
    }
