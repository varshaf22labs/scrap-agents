from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from trace_utils import TraceRecorder


class LLMClientError(RuntimeError):
    pass


@dataclass
class LLMClient:
    base_url: str
    provider: str = "openai"
    api_key: str | None = None
    model: str = "default"
    reasoning_effort: str | None = None
    timeout_seconds: int = 60
    trace_recorder: TraceRecorder | None = None

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if self.provider.lower() == "gemini":
            return self._chat_gemini(messages, temperature=temperature)
        return self._chat_openai(messages, temperature=temperature)

    def _supports_temperature(self) -> bool:
        model = self.model.lower()
        if self.provider.lower() == "gemini":
            return True
        if model.startswith("gpt-5"):
            return False
        return True

    def _chat_openai(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        url = self._chat_url()
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self._supports_temperature():
            payload["temperature"] = temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise LLMClientError(f"LLM request failed: {response.status_code} {response.text[:500]}")

        data = response.json()
        content: str | None = None
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        elif "output_text" in data:
            content = data["output_text"]
        elif "content" in data:
            content = data["content"]
        else:
            raise LLMClientError(f"Unexpected LLM response shape: {json.dumps(data)[:500]}")

        if self.trace_recorder:
            self.trace_recorder.record(
                "llm",
                "chat",
                {
                    "provider": self.provider,
                    "url": url,
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if self._supports_temperature() else None,
                },
                {"response": content, "raw_keys": sorted(data.keys())},
                note="LLM completion call",
            )
        return content

    def _chat_gemini(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if not self.api_key:
            raise LLMClientError("Gemini API key is required")
        base = self.base_url.rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            text = message.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        response = requests.post(
            f"{url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise LLMClientError(f"Gemini request failed: {response.status_code} {response.text[:500]}")
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMClientError(f"Unexpected Gemini response shape: {json.dumps(data)[:500]}")
        candidate = candidates[0]
        content = ""
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                content += part["text"]
        if not content:
            raise LLMClientError(f"Gemini response did not contain text: {json.dumps(data)[:500]}")
        if self.trace_recorder:
            self.trace_recorder.record(
                "llm",
                "chat",
                {"provider": self.provider, "url": url, "model": self.model, "messages": messages, "temperature": temperature},
                {"response": content, "raw_keys": sorted(data.keys())},
                note="Gemini completion call",
            )
        return content

    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start : end + 1])
