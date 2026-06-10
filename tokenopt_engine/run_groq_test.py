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

prompt = input("Enter your prompt/question: ").strip()
print("Enter context chunks. Submit an empty line to finish:")
chunks = []
while True:
    line = input("chunk> ").strip()
    if not line:
        break
    chunks.append(line)

optimizer = TokenOptimizer(TokenOptimizationConfig(
    mode="balanced",
    model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    max_input_tokens=8000,
    debug=True,
))

optimized = optimizer.optimize(prompt=prompt, context=chunks, query=prompt)
print("\n=== OPTIMIZATION METRICS ===")
print(json.dumps(optimized.metrics.to_dict(), indent=2, ensure_ascii=False))

client = GroqClient()
response = client.chat(optimized.optimized_prompt, max_tokens=1000)
print("\n=== RESPONSE ===")
print(response["content"])
print("\n=== GROQ USAGE ===")
print(json.dumps(response.get("usage", {}), indent=2, ensure_ascii=False))
