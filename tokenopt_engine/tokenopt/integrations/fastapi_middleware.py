# Optional helper. This file intentionally avoids importing FastAPI so the package remains dependency-light.
# Use the function below inside your own FastAPI route.

from typing import Any, Dict
from ..core.optimizer import TokenOptimizer


def optimize_request_payload(payload: Dict[str, Any], optimizer: TokenOptimizer) -> Dict[str, Any]:
    result = optimizer.optimize(
        prompt=payload.get("prompt", ""),
        context=payload.get("context"),
        query=payload.get("query") or payload.get("prompt", ""),
    )
    payload = dict(payload)
    payload["prompt"] = result.optimized_prompt
    payload["optimization_metrics"] = result.metrics.to_dict()
    return payload
