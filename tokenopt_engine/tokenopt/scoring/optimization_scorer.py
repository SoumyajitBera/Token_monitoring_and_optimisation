from ..math_utils import clamp


def optimization_success_score(retention_score: float, cost_reduction_score: float, latency_improvement_score: float = 0.0) -> float:
    return clamp(0.60 * retention_score + 0.30 * cost_reduction_score + 0.10 * latency_improvement_score)
