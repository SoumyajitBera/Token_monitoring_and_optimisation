from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from .token_utils import approx_tokens


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    real_api_call: bool


class GroqClient:
    def __init__(self, real: bool = False, model: str | None = None, call_delay: int = 0):
        load_dotenv()
        self.real = real
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.call_delay = max(0, int(call_delay))
        self.calls_made = 0
        if self.real:
            print("=" * 90)
            print("GROQ INITIALIZATION")
            print("=" * 90)
            print(f"GROQ_API_KEY loaded: {bool(self.api_key)}")
            print(f"GROQ_MODEL: {self.model}")
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY not found. Put it in project-root .env file.")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 220, temperature: float = 0.0) -> LLMResponse:
        input_tokens = approx_tokens(system_prompt + "\n" + user_prompt)
        if not self.real:
            text = self._mock_response(user_prompt)
            return LLMResponse(text, input_tokens, approx_tokens(text), input_tokens + approx_tokens(text), False)

        if self.calls_made > 0 and self.call_delay > 0:
            print(f"Sleeping {self.call_delay}s before next Groq call to reduce TPM/rate-limit risk...")
            time.sleep(self.call_delay)

        print("-" * 90)
        print(f"Sending request to Groq | model={self.model} | approx_input_tokens={input_tokens} | max_output={max_tokens}")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 429:
            print("Groq 429 rate limit hit. Waiting 90s once, then retrying...")
            time.sleep(90)
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Groq API call failed: HTTP {resp.status_code} | {resp.text[:1000]}") from exc
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens") or input_tokens)
        out_tok = int(usage.get("completion_tokens") or approx_tokens(text))
        self.calls_made += 1
        print(f"Groq response received | input={in_tok} | output={out_tok} | total={in_tok + out_tok}")
        return LLMResponse(text, in_tok, out_tok, in_tok + out_tok, True)

    def _mock_response(self, user_prompt: str) -> str:
        return "final_score: 8.0\nshortlist_decision: Yes\nreasons: ['Mock output only']\nconcerns: ['Run --real for Groq']"
