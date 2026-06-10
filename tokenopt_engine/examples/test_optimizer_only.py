import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from tokenopt import TokenOptimizer, TokenOptimizationConfig

prompt = """
Please kindly analyze the following situation. I would like to know the key decision.
Do not approve the request unless all required evidence is present.
The claim amount is ₹12500 and the policy code is ABC-2025.
The policy code is ABC-2025 and the claim amount is ₹12500.
"""

context = [
    "The request can be approved only if ECG evidence and symptom notes are both present.",
    "The request can be approved only if ECG evidence and symptom notes are both present.",
    "Unrelated marketing content about office cafeteria and travel reimbursement.",
    "Policy ABC-2025 requires prior clinical evidence before approval.",
]

optimizer = TokenOptimizer(TokenOptimizationConfig(mode="balanced", max_input_tokens=120, debug=True))
result = optimizer.optimize(prompt=prompt, context=context, query="Should the claim be approved under policy ABC-2025?")

print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
