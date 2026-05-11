from __future__ import annotations

import json
import re
import time
from typing import Any

from model.factory import SimpleChatModel, chat_model
from utils.logger_handler import log_stage, logger


class LLMCallError(RuntimeError):
    pass


def using_real_llm() -> bool:
    return not isinstance(chat_model, SimpleChatModel)


def invoke_llm(prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    with log_stage("llm.invoke", prompt_chars=len(prompt), model=chat_model.__class__.__name__) as stage:
        if not using_real_llm():
            raise LLMCallError("chat_model is SimpleChatModel fallback, not a real LLM provider.")
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = chat_model.invoke(prompt)
                content = getattr(response, "content", response)
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                content = str(content).strip()
                if not content:
                    raise LLMCallError("LLM returned empty content.")
                stage.add_done_fields(response_chars=len(content), attempt=attempt)
                return content
            except LLMCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(f"LLM invoke attempt {attempt}/{max_retries} failed: {exc}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        raise LLMCallError(f"LLM invoke failed after {max_retries} attempts: {last_exc}") from last_exc


def _repair_json_string(text: str) -> str:
    """Attempt basic repairs on malformed JSON output from LLM."""
    cleaned = text.strip()
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # Remove inline comments like // or #
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    # Unescape double-escaped quotes
    cleaned = cleaned.replace("\\\"", "\"")
    # Try to close unclosed braces
    open_braces = cleaned.count("{") - cleaned.count("}")
    open_brackets = cleaned.count("[") - cleaned.count("]")
    if open_braces > 0:
        cleaned += "}" * open_braces
    if open_brackets > 0:
        cleaned += "]" * open_brackets
    return cleaned


def extract_json_object(text: str, max_repair_attempts: int = 2) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    for attempt in range(max_repair_attempts + 1):
        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise LLMCallError("LLM JSON output is not an object.")
            return data
        except json.JSONDecodeError as exc:
            if attempt < max_repair_attempts:
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {exc}, attempting repair...")
                cleaned = _repair_json_string(cleaned)
            else:
                raise LLMCallError(f"Failed to parse JSON from LLM output after {max_repair_attempts + 1} attempts: {text[:500]}") from exc


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)