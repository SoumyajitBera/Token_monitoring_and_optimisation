from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict
from .agents import default_agents
from .llm_client import GroqClient
from .optimizer import AggressiveSafeOptimizer
from .costing import estimate_cost_usd
from .quality import compare_outputs


def compare_baseline_vs_optimized(
    context: str,
    real: bool,
    api_strategy: str,
    workflow_delay: int,
    call_delay: int,
    max_output_tokens: int,
    optimization_level: str,
    print_context_preview: bool,
) -> dict:
    agents = default_agents()
    baseline_llm = GroqClient(real=real, call_delay=call_delay)
    optimized_llm = GroqClient(real=real, call_delay=call_delay)
    optimizer = AggressiveSafeOptimizer(level=optimization_level)

    baseline_outputs = []
    optimized_outputs = []
    agent_reports = []
    baseline_context = context
    optimized_context = context

    if api_strategy in {"both", "baseline-only"}:
        print("=" * 90)
        print("BASELINE WORKFLOW - REAL RAW CONTEXT")
        print("=" * 90)
        for i, agent in enumerate(agents, start=1):
            print(f"[{i}/{len(agents)}] {agent.name}")
            resp = agent.run_baseline(baseline_llm, baseline_context, max_output_tokens)
            print(resp.text[:1200])
            baseline_outputs.append((agent.name, resp))
            baseline_context += f"\n\n[{agent.name} OUTPUT]\n{resp.text}"

    if api_strategy == "both" and workflow_delay > 0:
        print("=" * 90)
        print(f"WAITING {workflow_delay} SECONDS BETWEEN BASELINE AND OPTIMIZED WORKFLOW")
        print("=" * 90)
        time.sleep(workflow_delay)

    if api_strategy in {"both", "optimized-only"}:
        print("=" * 90)
        print("OPTIMIZED WORKFLOW - AGGRESSIVE SAFE CONTEXT")
        print("=" * 90)
        for i, agent in enumerate(agents, start=1):
            print(f"[{i}/{len(agents)}] {agent.name}")
            resp, opt = agent.run_optimized(optimized_llm, optimized_context, optimizer, max_output_tokens)
            if print_context_preview:
                print("ORIGINAL_CONTEXT_PREVIEW:")
                print(opt.original_text[:900])
                print("OPTIMIZED_CONTEXT_PREVIEW:")
                print(opt.optimized_text[:900])
                print(f"Context reduction: {opt.reduction_pct:.2f}%")
            print(resp.text[:1200])
            optimized_outputs.append((agent.name, resp, opt))
            # Critical: pass compressed state forward, not full raw history.
            optimized_context = opt.optimized_text + f"\n\n[{agent.name} OUTPUT]\n{resp.text}"

    base_in = sum(r.input_tokens for _, r in baseline_outputs)
    base_out = sum(r.output_tokens for _, r in baseline_outputs)
    opt_in = sum(r.input_tokens for _, r, _ in optimized_outputs)
    opt_out = sum(r.output_tokens for _, r, _ in optimized_outputs)
    base_total = base_in + base_out
    opt_total = opt_in + opt_out
    reduction = 0.0 if base_total == 0 else (1 - opt_total / base_total) * 100
    base_cost = estimate_cost_usd(base_in, base_out)
    opt_cost = estimate_cost_usd(opt_in, opt_out)
    cost_reduction = 0.0 if base_cost == 0 else (1 - opt_cost / base_cost) * 100

    final_baseline = baseline_outputs[-1][1].text if baseline_outputs else ""
    final_optimized = optimized_outputs[-1][1].text if optimized_outputs else ""
    quality = compare_outputs(final_baseline, final_optimized) if final_baseline and final_optimized else None

    result = {
        "execution_mode": "REAL_GROQ_API_FORCED" if real else "MOCK",
        "api_strategy": api_strategy,
        "agent_count": len(agents),
        "workflow_delay_seconds": workflow_delay,
        "call_delay_seconds": call_delay,
        "optimization_level": optimization_level,
        "external_api_calls_made": baseline_llm.calls_made + optimized_llm.calls_made,
        "baseline_total_input_tokens": base_in,
        "baseline_total_output_tokens": base_out,
        "baseline_total_tokens": base_total,
        "optimized_total_input_tokens": opt_in,
        "optimized_total_output_tokens": opt_out,
        "optimized_total_tokens": opt_total,
        "token_reduction_pct": reduction,
        "baseline_cost_usd": base_cost,
        "optimized_cost_usd": opt_cost,
        "savings_usd": base_cost - opt_cost,
        "cost_reduction_pct": cost_reduction,
        "quality": asdict(quality) if quality else None,
        "baseline_final_answer": final_baseline,
        "optimized_final_answer": final_optimized,
        "optimized_context_reductions": [
            {
                "agent": name,
                "context_reduction_pct": opt.reduction_pct,
                "original_tokens": opt.original_tokens,
                "optimized_tokens": opt.optimized_tokens,
                "retention_guard_passed": opt.retention_guard_passed,
                "missing_protected_terms": opt.missing_protected_terms,
                "protected_terms_count": len(opt.protected_terms),
            }
            for name, _, opt in optimized_outputs
        ],
    }
    _save_outputs(result)
    _print_summary(result)
    return result


def _save_outputs(result: dict) -> None:
    Path("benchmark_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# ProdSync Agentic Token Optimization Benchmark", ""]
    for k, v in result.items():
        if k not in {"baseline_final_answer", "optimized_final_answer", "optimized_context_reductions"}:
            md.append(f"- **{k}**: {v}")
    md.extend(["", "## Optimized Context Reductions", ""])
    for item in result["optimized_context_reductions"]:
        md.append(f"- {item['agent']}: {item['context_reduction_pct']:.2f}% ({item['original_tokens']} -> {item['optimized_tokens']}); guard={item.get('retention_guard_passed')}; missing={item.get('missing_protected_terms')}")
    md.extend(["", "## Baseline Final Answer", "", "```", result.get("baseline_final_answer", ""), "```"])
    md.extend(["", "## Optimized Final Answer", "", "```", result.get("optimized_final_answer", ""), "```"])
    Path("benchmark_report.md").write_text("\n".join(md), encoding="utf-8")


def _print_summary(r: dict) -> None:
    print("=" * 90)
    print("TOKEN ANALYTICS")
    print("=" * 90)
    print(f"Baseline input tokens:   {r['baseline_total_input_tokens']}")
    print(f"Baseline output tokens:  {r['baseline_total_output_tokens']}")
    print(f"Baseline total tokens:   {r['baseline_total_tokens']}")
    print(f"Optimized input tokens:  {r['optimized_total_input_tokens']}")
    print(f"Optimized output tokens: {r['optimized_total_output_tokens']}")
    print(f"Optimized total tokens:  {r['optimized_total_tokens']}")
    print(f"Token reduction:         {r['token_reduction_pct']:.2f}%")
    print("=" * 90)
    print("COST ANALYTICS")
    print("=" * 90)
    print(f"Baseline cost USD:  {r['baseline_cost_usd']:.8f}")
    print(f"Optimized cost USD: {r['optimized_cost_usd']:.8f}")
    print(f"Savings USD:        {r['savings_usd']:.8f}")
    print(f"Cost reduction:     {r['cost_reduction_pct']:.2f}%")
    if r.get("quality"):
        print("=" * 90)
        print("QUALITY PRESERVATION")
        print("=" * 90)
        q = r["quality"]
        print(f"Decision preserved:     {q['decision_preserved']}")
        print(f"Score delta:            {q['score_delta']}")
        print(f"Keyword/entity retention:{q['keyword_retention_pct']:.2f}%")
        print(f"Entity retention:       {q.get('entity_retention_pct', q['keyword_retention_pct']):.2f}%")
        print(f"Concern retention:      {q.get('concern_retention_pct', 0):.2f}%")
        print(f"Reason overlap:         {q.get('reason_overlap_pct', 0):.2f}%")
        print(f"Output lexical sim:     {q['output_similarity']:.3f}")
        print(f"Overall retention est.: {q['overall_retention_pct']:.2f}%")
        print(f"Missing entities:       {q.get('missing_entities', [])}")
        print(f"Missing concerns:       {q.get('missing_concerns', [])}")
        print(f"Baseline decision:      {q.get('baseline_decision')}")
        print(f"Optimized decision:     {q.get('optimized_decision')}")
        print(f"Baseline score:         {q.get('baseline_score')}")
        print(f"Optimized score:        {q.get('optimized_score')}")
    if r.get("optimized_context_reductions"):
        print("=" * 90)
        print("EVIDENCE GUARDRAIL STATUS")
        print("=" * 90)
        for item in r["optimized_context_reductions"]:
            print(f"{item['agent']}: guard={item.get('retention_guard_passed')} | protected_terms={item.get('protected_terms_count')} | missing={item.get('missing_protected_terms')}")
    print("=" * 90)
    print("FINAL RECRUITER DECISION OUTPUT")
    print("=" * 90)
    print("BASELINE FINAL ANSWER")
    print("-" * 90)
    print(r.get("baseline_final_answer", ""))
    print("OPTIMIZED FINAL ANSWER")
    print("-" * 90)
    print(r.get("optimized_final_answer", ""))
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)
    print("benchmark_result.json")
    print("benchmark_report.md")
