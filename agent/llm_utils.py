from __future__ import annotations

import json
import re
from typing import Any

from model.factory import SimpleChatModel, chat_model


class LLMCallError(RuntimeError):
    pass


def using_real_llm() -> bool:
    return not isinstance(chat_model, SimpleChatModel)


def invoke_llm(prompt: str) -> str:
    if not using_real_llm():
        raise LLMCallError("chat_model is SimpleChatModel fallback, not a real LLM provider.")
    try:
        response = chat_model.invoke(prompt)
    except Exception as exc:
        raise LLMCallError(str(exc)) from exc

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    content = str(content).strip()
    if not content:
        raise LLMCallError("LLM returned empty content.")
    return content


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMCallError(f"Failed to parse JSON from LLM output: {text[:500]}") from exc
    if not isinstance(data, dict):
        raise LLMCallError("LLM JSON output is not an object.")
    return data


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
