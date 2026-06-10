from typing import Dict


def estimate_cost(input_tokens: int, expected_output_tokens: int, input_price_per_million: float, output_price_per_million: float) -> float:
    return (input_tokens / 1_000_000.0) * input_price_per_million + (expected_output_tokens / 1_000_000.0) * output_price_per_million


def cost_metrics(original_tokens: int, optimized_tokens: int, expected_output_tokens: int, input_price_per_million: float, output_price_per_million: float) -> Dict[str, float]:
    before = estimate_cost(original_tokens, expected_output_tokens, input_price_per_million, output_price_per_million)
    after = estimate_cost(optimized_tokens, expected_output_tokens, input_price_per_million, output_price_per_million)
    saved = max(0.0, before - after)
    reduction_score = saved / before if before > 0 else 0.0
    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_pct = (tokens_saved / original_tokens * 100.0) if original_tokens > 0 else 0.0
    return {
        "estimated_cost_before": before,
        "estimated_cost_after": after,
        "estimated_savings": saved,
        "cost_reduction_score": reduction_score,
        "tokens_saved": tokens_saved,
        "reduction_percentage": reduction_pct,
    }
