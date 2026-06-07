from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_token_optimizer.data_factory import build_context
from agentic_token_optimizer.benchmark import compare_baseline_vs_optimized


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="Force real Groq API calls using GROQ_API_KEY from .env")
    p.add_argument("--api-strategy", choices=["both", "baseline-only", "optimized-only"], default="both")
    p.add_argument("--workflow-delay", type=int, default=90, help="Sleep between baseline workflow and optimized workflow")
    p.add_argument("--call-delay", type=int, default=8, help="Sleep between individual Groq calls")
    p.add_argument("--max-output-tokens", type=int, default=200)
    p.add_argument("--sample-profile", choices=["medium", "large"], default="large")
    p.add_argument("--optimization-level", choices=["moderate", "aggressive", "guarded"], default="guarded")
    p.add_argument("--print-context-preview", action="store_true")
    args = p.parse_args()

    context = build_context(args.sample_profile)
    compare_baseline_vs_optimized(
        context=context,
        real=args.real,
        api_strategy=args.api_strategy,
        workflow_delay=args.workflow_delay,
        call_delay=args.call_delay,
        max_output_tokens=args.max_output_tokens,
        optimization_level=args.optimization_level,
        print_context_preview=args.print_context_preview,
    )


if __name__ == "__main__":
    main()
