from typing import Dict, List, Optional, Sequence, Union

from ..compression.safe_normalizer import safe_normalize
from ..compression.sentence_deduplicator import deduplicate_sentences
from ..retrieval.diversity_selector import greedy_diversity_select
from ..retrieval.knapsack_selector import select_chunks_knapsack
from ..retrieval.relevance_filter import filter_relevant_chunks
from ..scoring.cost_scorer import cost_metrics
from ..scoring.optimization_scorer import optimization_success_score
from ..tokenization import count_tokens
from ..validation.retention_validator import RetentionValidator
from .schemas import OptimizationMetrics, OptimizationResult, TokenOptimizationConfig


def _join_prompt(prompt: str, selected_chunks: Sequence[str]) -> str:
    if selected_chunks:
        return (prompt or "").strip() + "\n\n[OPTIMIZED_CONTEXT]\n" + "\n\n".join(c.strip() for c in selected_chunks if c and c.strip())
    return (prompt or "").strip()


class TokenOptimizer:
    """Reusable token optimization middleware for LLM calls.

    It performs safe normalization, duplicate pruning, relevance filtering,
    token-budget chunk selection, retention scoring, cost scoring, and rollback.
    """

    def __init__(self, config: Optional[TokenOptimizationConfig] = None, **kwargs):
        if config is None:
            config = TokenOptimizationConfig(**kwargs)
        elif kwargs:
            data = config.__dict__.copy()
            data.update(kwargs)
            config = TokenOptimizationConfig(**data)
        self.config = config.resolved()
        self.validator = RetentionValidator()

    def optimize(
        self,
        prompt: str,
        context: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
    ) -> OptimizationResult:
        cfg = self.config
        max_tokens = max_input_tokens or cfg.max_input_tokens
        debug: Dict[str, object] = {"mode": cfg.mode, "max_input_tokens": max_tokens}

        if isinstance(context, str):
            chunks = [context]
        else:
            chunks = context or []

        original_full_prompt = _join_prompt(prompt, chunks)
        original_tokens = count_tokens(original_full_prompt)

        working_prompt = prompt or ""
        if cfg.enable_safe_normalization:
            working_prompt, norm_info = safe_normalize(working_prompt)
            debug["safe_normalization"] = norm_info

        if cfg.enable_sentence_deduplication:
            working_prompt, dedup_info = deduplicate_sentences(
                working_prompt,
                threshold=cfg.duplicate_similarity_threshold or 0.90,
            )
            debug["prompt_sentence_deduplication"] = dedup_info

        selected_chunk_texts: List[str] = []
        removed_chunk_texts: List[str] = []

        if chunks:
            candidate_chunks, relevance_removed, rel_info = filter_relevant_chunks(
                query or prompt,
                chunks,
                threshold=cfg.relevance_similarity_threshold or 0.15,
            )
            debug["relevance_filtering"] = rel_info
            removed_chunk_texts.extend([c.text for c in relevance_removed])

            diverse_chunks, diversity_removed, div_info = greedy_diversity_select(
                candidate_chunks,
                max_similarity=cfg.duplicate_similarity_threshold or 0.90,
            )
            debug["diversity_selection"] = div_info
            removed_chunk_texts.extend([c.text for c in diversity_removed])

            prompt_budget = max(1, max_tokens - count_tokens(working_prompt))
            if cfg.enable_knapsack_selection:
                selected_chunks, knap_removed, knap_info = select_chunks_knapsack(diverse_chunks, prompt_budget)
                debug["knapsack_selection"] = knap_info
                removed_chunk_texts.extend([c.text for c in knap_removed])
            else:
                selected_chunks = []
                used = 0
                for c in diverse_chunks:
                    if used + c.token_cost <= prompt_budget:
                        selected_chunks.append(c)
                        used += c.token_cost
                    else:
                        removed_chunk_texts.append(c.text)
                debug["greedy_budget_selection"] = {"used_tokens": used, "budget_tokens": prompt_budget}
            selected_chunk_texts = [c.text for c in selected_chunks]

        optimized_candidate = _join_prompt(working_prompt, selected_chunk_texts)

        # If no chunks and prompt is still above budget, safely trim by sentences from the end.
        # Retention validator + rollback protects correctness.
        if count_tokens(optimized_candidate) > max_tokens:
            from ..tokenization import split_sentences
            sentences = split_sentences(working_prompt)
            kept = []
            for s in sentences:
                candidate = _join_prompt("\n".join(kept + [s]), selected_chunk_texts)
                if count_tokens(candidate) <= max_tokens:
                    kept.append(s)
                else:
                    removed_chunk_texts.append(s)
            optimized_candidate = _join_prompt("\n".join(kept), selected_chunk_texts)
            debug["sentence_budget_trim"] = {"kept_sentences": len(kept), "original_sentences": len(sentences)}

        # Retention rescue: if optimization removed too much, add removed chunks back
        # while budget allows. This is the guardrail that prevents silent information loss.
        retention = self.validator.score(original_full_prompt, optimized_candidate)
        if chunks and retention["retention_score"] < (cfg.min_retention_score or 0.90):
            # Unique rescue pool avoids adding duplicate chunks just to satisfy brittle coverage.
            rescue_pool = []
            seen_rescue = set(selected_chunk_texts)
            for x in removed_chunk_texts:
                if x and x not in seen_rescue:
                    rescue_pool.append(x)
                    seen_rescue.add(x)
            # Prefer chunks with constraints/numbers/entities and higher query similarity.
            from ..extractors import extract_constraints, extract_numbers, extract_entities, extract_codes
            from ..math_utils import cosine_text
            def rescue_score(t: str) -> float:
                return (
                    3.0 * len(extract_numbers(t))
                    + 2.0 * len(extract_codes(t))
                    + 1.5 * len(extract_constraints(t))
                    + 0.5 * len(extract_entities(t))
                    + cosine_text(query or prompt, t)
                )
            rescue_pool.sort(key=rescue_score, reverse=True)
            rescue_added = []
            for txt in rescue_pool:
                candidate_chunks = selected_chunk_texts + rescue_added + [txt]
                candidate_prompt = _join_prompt(working_prompt, candidate_chunks)
                if count_tokens(candidate_prompt) <= max_tokens:
                    candidate_retention = self.validator.score(original_full_prompt, candidate_prompt)
                    if candidate_retention["retention_score"] >= retention["retention_score"] + 0.002:
                        rescue_added.append(txt)
                        optimized_candidate = candidate_prompt
                        retention = candidate_retention
                    if retention["retention_score"] >= (cfg.min_retention_score or 0.90):
                        break
            selected_chunk_texts.extend(rescue_added)
            debug["retention_rescue"] = {"added_chunks": len(rescue_added), "retention_after_rescue": retention["retention_score"]}

        optimized_tokens_candidate = count_tokens(optimized_candidate)

        costs = cost_metrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens_candidate,
            expected_output_tokens=cfg.expected_output_tokens,
            input_price_per_million=cfg.input_price_per_million_tokens,
            output_price_per_million=cfg.output_price_per_million_tokens,
        )

        success_score = optimization_success_score(
            retention_score=retention["retention_score"],
            cost_reduction_score=costs["cost_reduction_score"],
            latency_improvement_score=costs["cost_reduction_score"],
        )

        status = "accepted"
        reason = None
        final_prompt = optimized_candidate
        final_tokens = optimized_tokens_candidate

        if cfg.enable_rollback and retention["retention_score"] < (cfg.min_retention_score or 0.90):
            status = "rolled_back"
            reason = "retention_score_below_threshold"
            final_prompt = original_full_prompt
            final_tokens = original_tokens
            costs = cost_metrics(
                original_tokens=original_tokens,
                optimized_tokens=final_tokens,
                expected_output_tokens=cfg.expected_output_tokens,
                input_price_per_million=cfg.input_price_per_million_tokens,
                output_price_per_million=cfg.output_price_per_million_tokens,
            )
            success_score = optimization_success_score(retention["retention_score"], costs["cost_reduction_score"], 0.0)


        def _risk_label(ret: Dict[str, object], reduction_pct: float, stat: str) -> str:
            if stat == "rolled_back":
                return "ROLLED_BACK"
            cr = float(ret.get("constraint_retention", 0.0) or 0.0)
            nr = float(ret.get("numeric_retention", 0.0) or 0.0)
            er = float(ret.get("entity_retention", 0.0) or 0.0)
            rs = float(ret.get("retention_score", 0.0) or 0.0)
            if rs >= 0.94 and cr >= 0.88 and nr >= 0.98 and er >= 0.90 and reduction_pct <= 30:
                return "LOW"
            if rs >= 0.88 and cr >= 0.72 and nr >= 0.92 and er >= 0.80:
                return "MEDIUM"
            return "HIGH"

        risk_label = _risk_label(retention, costs["reduction_percentage"], status)

        metrics = OptimizationMetrics(
            original_tokens=original_tokens,
            optimized_tokens=final_tokens,
            tokens_saved=costs["tokens_saved"],
            reduction_percentage=round(costs["reduction_percentage"], 4),
            estimated_cost_before=round(costs["estimated_cost_before"], 8),
            estimated_cost_after=round(costs["estimated_cost_after"], 8),
            estimated_savings=round(costs["estimated_savings"], 8),
            retention_score=round(retention["retention_score"], 4),
            entity_retention=round(retention["entity_retention"], 4),
            numeric_retention=round(retention["numeric_retention"], 4),
            constraint_retention=round(retention["constraint_retention"], 4),
            semantic_similarity=round(retention["semantic_similarity"], 4),
            keyword_coverage=round(retention["keyword_coverage"], 4),
            cost_reduction_score=round(costs["cost_reduction_score"], 4),
            optimization_success_score=round(success_score, 4),
            risk_label=risk_label,
            status=status,
            reason=reason,
        )

        debug["retention_details"] = retention
        return OptimizationResult(
            original_prompt=original_full_prompt,
            optimized_prompt=final_prompt,
            metrics=metrics,
            selected_chunks=selected_chunk_texts,
            removed_chunks=removed_chunk_texts,
            debug_info=debug if cfg.debug else {},
        )

    def evaluate(self, original_prompt: str, optimized_prompt: str) -> Dict[str, float]:
        return self.validator.score(original_prompt, optimized_prompt)
