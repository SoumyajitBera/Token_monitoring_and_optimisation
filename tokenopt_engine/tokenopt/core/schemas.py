from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    value: float = 0.0
    token_cost: int = 0


@dataclass
class TokenOptimizationConfig:
    mode: str = "balanced"  # conservative | balanced | aggressive
    model_name: str = "llama-3.3-70b-versatile"
    max_input_tokens: int = 8000
    min_retention_score: Optional[float] = None
    enable_safe_normalization: bool = True
    enable_sentence_deduplication: bool = True
    enable_relevance_filtering: bool = True
    enable_knapsack_selection: bool = True
    enable_rollback: bool = True
    debug: bool = False
    risk_labeling_enabled: bool = True

    # v1.5 schema-aware optimization controls. These are generic: any
    # application can pass protected contract text / required field names.
    schema_strict: bool = False
    protected_texts: List[str] = field(default_factory=list)
    schema_critical_terms: List[str] = field(default_factory=list)

    filler_threshold: float = 0.015
    duplicate_similarity_threshold: Optional[float] = None
    relevance_similarity_threshold: Optional[float] = None

    input_price_per_million_tokens: float = 0.59
    output_price_per_million_tokens: float = 0.79
    expected_output_tokens: int = 1000

    def resolved(self) -> "TokenOptimizationConfig":
        cfg = TokenOptimizationConfig(**asdict(self))
        mode = (cfg.mode or "balanced").lower().strip()
        cfg.mode = mode
        if mode == "conservative":
            cfg.min_retention_score = cfg.min_retention_score if cfg.min_retention_score is not None else 0.95
            cfg.duplicate_similarity_threshold = cfg.duplicate_similarity_threshold if cfg.duplicate_similarity_threshold is not None else 0.94
            cfg.relevance_similarity_threshold = cfg.relevance_similarity_threshold if cfg.relevance_similarity_threshold is not None else 0.12
        elif mode == "aggressive":
            cfg.min_retention_score = cfg.min_retention_score if cfg.min_retention_score is not None else 0.82
            cfg.duplicate_similarity_threshold = cfg.duplicate_similarity_threshold if cfg.duplicate_similarity_threshold is not None else 0.86
            cfg.relevance_similarity_threshold = cfg.relevance_similarity_threshold if cfg.relevance_similarity_threshold is not None else 0.20
        else:
            cfg.min_retention_score = cfg.min_retention_score if cfg.min_retention_score is not None else 0.88
            cfg.duplicate_similarity_threshold = cfg.duplicate_similarity_threshold if cfg.duplicate_similarity_threshold is not None else 0.90
            cfg.relevance_similarity_threshold = cfg.relevance_similarity_threshold if cfg.relevance_similarity_threshold is not None else 0.15
        if cfg.schema_strict:
            # Schema-critical workflows should not use ultra-aggressive pruning.
            if cfg.mode == "aggressive":
                cfg.mode = "balanced"
                cfg.duplicate_similarity_threshold = max(cfg.duplicate_similarity_threshold or 0.90, 0.90)
                cfg.relevance_similarity_threshold = min(cfg.relevance_similarity_threshold or 0.15, 0.15)
            cfg.min_retention_score = max(float(cfg.min_retention_score or 0.0), 0.88)
        return cfg


@dataclass
class OptimizationMetrics:
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    reduction_percentage: float
    estimated_cost_before: float
    estimated_cost_after: float
    estimated_savings: float
    retention_score: float
    entity_retention: float
    numeric_retention: float
    constraint_retention: float
    semantic_similarity: float
    keyword_coverage: float
    cost_reduction_score: float
    optimization_success_score: float
    risk_label: str
    status: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    metrics: OptimizationMetrics
    selected_chunks: List[str] = field(default_factory=list)
    removed_chunks: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "metrics": self.metrics.to_dict(),
            "selected_chunks": self.selected_chunks,
            "removed_chunks": self.removed_chunks,
            "debug_info": self.debug_info,
        }
