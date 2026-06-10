import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional


class GroqClient:
    """Minimal Groq chat client using only Python standard library.

    Important:
    Some Groq edge/WAF paths reject Python urllib requests when no User-Agent
    is provided and return HTTP 403 with error code 1010. This client sends a
    normal application User-Agent and gives actionable diagnostics.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 90,
        base_url: Optional[str] = None,
        user_agent: str = "TokenOptEngine/1.1 (+https://api.groq.com)",
    ):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
        self.model = (model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).strip().strip('"').strip("'")
        self.timeout = timeout
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.user_agent = user_agent
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in environment or .env file.")
        if not self.model:
            raise ValueError("GROQ_MODEL is missing. Set it in environment or .env file.")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    def _post_json(self, endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(self._format_http_error(e.code, body)) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Groq connection error: {e}. Check internet/proxy/VPN/firewall settings.") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Groq returned non-JSON response from {url}: {e}") from e

    def _get_json(self, endpoint: str) -> Dict[str, object]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(self._format_http_error(e.code, body)) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Groq connection error: {e}. Check internet/proxy/VPN/firewall settings.") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Groq returned non-JSON response from {url}: {e}") from e

    def _format_http_error(self, code: int, body: str) -> str:
        body_clean = (body or "").strip()
        hints = []
        if code == 401:
            hints.append("Invalid or expired GROQ_API_KEY.")
        if code == 403:
            hints.append("Forbidden. Possible causes: missing User-Agent/WAF block, restricted model permission, project/org permission, region/network/VPN block, or disabled billing/access.")
            if "1010" in body_clean:
                hints.append("Groq/Cloudflare 1010 is commonly triggered by blocked request fingerprinting. This client now sends User-Agent; if it still fails, test from another network or install the official Groq SDK.")
        if code == 404:
            hints.append("Endpoint or model not found. Check GROQ_MODEL and GROQ_BASE_URL.")
        if code == 429:
            hints.append("Rate limit hit. Reduce request rate or check Groq limits.")
        if code >= 500:
            hints.append("Provider/server-side issue. Retry later.")
        hint_text = " ".join(hints)
        return f"Groq HTTP error {code}: {body_clean}. {hint_text}"

    def list_models(self) -> Dict[str, object]:
        """Return available model metadata for this API key/project."""
        return self._get_json("models")

    def health_check(self) -> Dict[str, object]:
        """Small diagnostic call that does not consume chat tokens."""
        models = self.list_models()
        model_ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        return {
            "ok": True,
            "configured_model": self.model,
            "model_visible": self.model in model_ids,
            "visible_model_count": len(model_ids),
            "sample_models": model_ids[:10],
        }

    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a precise and helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> Dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        parsed = self._post_json("chat/completions", payload)
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = parsed.get("usage", {})
        return {"content": content, "usage": usage, "raw": parsed}
