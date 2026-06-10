from typing import Dict, Tuple
from ..math_utils import shannon_entropy


def entropy_profile(text: str) -> Dict[str, float]:
    return {"shannon_entropy": shannon_entropy(text)}


def entropy_filter(text: str, min_entropy: float = 0.0) -> Tuple[str, Dict[str, float]]:
    """Placeholder-safe entropy hook.

    It currently does not delete tokens because blind entropy deletion can destroy meaning.
    The entropy score is exposed so you can later tune policies safely.
    """
    score = shannon_entropy(text)
    return text, {"shannon_entropy": score, "min_entropy": min_entropy, "changed": False}
