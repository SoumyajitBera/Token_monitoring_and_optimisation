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
    repaired: bool = False
    raw_text: str | None = None
    json_mode_used: bool = False
    fallback_used: bool = False
    schema_valid: bool | None = None
    repair_severity: str = "none"  # none | minor | major | fallback


class GroqClient:
    """Thin Groq REST client with deterministic settings and safe JSON fallback.

    Important fix:
    Groq JSON mode can fail with HTTP 400 `json_validate_failed` when max_tokens is
    too small to finish a valid JSON object. That was the error you hit. We now do
    NOT depend on provider JSON mode by default. The prompt asks for JSON and the
    local deterministic repair layer in agents.py normalizes/parses the output.
    """

    def __init__(self, real: bool = False, model: str | None = None, call_delay: int = 0, use_json_mode: bool = False):
        load_dotenv()
        self.real = real
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.call_delay = max(0, int(call_delay))
        # Default is False because strict JSON mode + low max_tokens causes HTTP 400.
        self.use_json_mode = use_json_mode or os.getenv("GROQ_JSON_MODE", "false").lower() in {"1", "true", "yes"}
        self.calls_made = 0
        if self.real:
            print("=" * 90)
            print("GROQ INITIALIZATION")
            print("=" * 90)
            print(f"GROQ_API_KEY loaded: {bool(self.api_key)}")
            print(f"GROQ_MODEL: {self.model}")
            print(f"GROQ_JSON_MODE: {self.use_json_mode}")
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY not found. Put it in project-root .env file.")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 260, temperature: float = 0.0) -> LLMResponse:
        input_tokens = approx_tokens(system_prompt + "\n" + user_prompt)
        if not self.real:
            text = self._mock_response(user_prompt)
            return LLMResponse(text, input_tokens, approx_tokens(text), input_tokens + approx_tokens(text), False)

        if self.calls_made > 0 and self.call_delay > 0:
            print(f"Sleeping {self.call_delay}s before next Groq call to reduce TPM/rate-limit risk...")
            time.sleep(self.call_delay)

        print("-" * 90)
        print(f"Sending request to Groq | model={self.model} | approx_input_tokens={input_tokens} | max_output={max_tokens}")

        payload = self._payload(system_prompt, user_prompt, max_tokens, temperature, json_mode=self.use_json_mode)
        resp = self._post(payload)
        fallback_used = False

        if resp.status_code == 400:
            body = resp.text.lower()
            # Main fix for your current error:
            # json_validate_failed / max completion tokens reached before valid JSON.
            if "json_validate_failed" in body or "failed to generate json" in body or "response_format" in body:
                print("Groq JSON-mode validation failed or output budget too small for strict JSON. Retrying once without response_format...")
                payload = self._payload(system_prompt, user_prompt, max_tokens, temperature, json_mode=False)
                resp = self._post(payload)
                fallback_used = True

        if resp.status_code == 429:
            print("Groq 429 rate limit hit. Waiting 90s once, then retrying...")
            time.sleep(90)
            resp = self._post(payload)

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Groq API call failed: HTTP {resp.status_code} | {resp.text[:1000]}") from exc

        data = resp.json()
        text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens") or input_tokens)
        out_tok = int(usage.get("completion_tokens") or approx_tokens(text))
        self.calls_made += 1
        print(f"Groq response received | input={in_tok} | output={out_tok} | total={in_tok + out_tok} | fallback={fallback_used}")
        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            real_api_call=True,
            raw_text=text,
            json_mode_used=self.use_json_mode and not fallback_used,
            fallback_used=fallback_used,
        )

    def _payload(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, json_mode: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict) -> requests.Response:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return requests.post(url, headers=headers, json=payload, timeout=90)

    def _mock_response(self, user_prompt: str) -> str:
        return '{"final_score":8.0,"shortlist_decision":"Shortlist","evidence_terms":["Mock output only"],"reasons":["Run --real for Groq"],"concerns":["Run --real for Groq"],"interview_plan":["N/A"]}'
