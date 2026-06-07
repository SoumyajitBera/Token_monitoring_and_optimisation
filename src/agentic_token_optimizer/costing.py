from __future__ import annotations

# Groq price values are configurable approximations. Adjust as needed.
DEFAULT_INPUT_PRICE_PER_MTOK = 0.59
DEFAULT_OUTPUT_PRICE_PER_MTOK = 0.79


def estimate_cost_usd(input_tokens: int, output_tokens: int, input_price: float = DEFAULT_INPUT_PRICE_PER_MTOK, output_price: float = DEFAULT_OUTPUT_PRICE_PER_MTOK) -> float:
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
