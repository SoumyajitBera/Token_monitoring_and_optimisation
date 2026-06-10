import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from pathlib import Path
from tokenopt import TokenOptimizer, TokenOptimizationConfig
from tokenopt.integrations import GroqClient


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

prompt = """
Please kindly answer the user question using the context. Do not ignore any numbers, dates, policy codes, or negation.
Question: Should the claim be approved? Give a short reason.
"""

context = [
    "Policy ABC-2025 says approval is allowed only if ECG evidence is present before claim submission.",
    "The claim amount is ₹12500. Submission date is 12 May 2026.",
    "The clinical note says symptoms were present, but ECG evidence was not attached.",
    "Random duplicated text: approval is allowed only if ECG evidence is present before claim submission.",
]

optimizer = TokenOptimizer(TokenOptimizationConfig(
    mode="balanced",
    model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    max_input_tokens=1000,
    debug=True,
))

optimized = optimizer.optimize(
    prompt=prompt,
    context=context,
    query="Should the claim be approved under ABC-2025?"
)

print("\n=== OPTIMIZATION METRICS ===")
print(json.dumps(optimized.metrics.to_dict(), indent=2, ensure_ascii=False))

client = GroqClient()

print("\n=== GROQ HEALTH CHECK ===")
try:
    print(json.dumps(client.health_check(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Health check failed: {e}")

response = client.chat(
    prompt=optimized.optimized_prompt,
    system_prompt="You are a strict decision assistant. Preserve policy constraints and numbers.",
    temperature=0.1,
    max_tokens=500,
)

print("\n=== GROQ RESPONSE ===")
print(response["content"])
print("\n=== GROQ USAGE ===")
print(json.dumps(response.get("usage", {}), indent=2, ensure_ascii=False))
