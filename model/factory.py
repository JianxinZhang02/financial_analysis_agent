from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from utils.config_handler import model_cof, rag_cof


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Any:
        pass


@dataclass
class SimpleMessage:
    content: str


class SimpleChatModel:
    """Deterministic fallback used when LangChain providers are unavailable."""

    def __init__(self, model: str = "local-rule-based"):
        self.model = model

    def invoke(self, prompt: Any) -> SimpleMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return SimpleMessage(
            content=(
                "当前环境未安装在线大模型依赖，已启用本地规则模型。"
                "安装 requirements.txt 后可接入真实 LLM。\n\n"
                f"输入摘要：{text[:500]}"
            )
        )


class SimpleEmbeddings:
    """Small hashing embedding fallback with the LangChain embedding interface."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_.%+-]+|[\u4e00-\u9fff]", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Any:
        provider = model_cof.get("chat_provider", rag_cof.get("chat_provider", "dashscope"))
        model_name = model_cof.get("chat_model_name", rag_cof.get("chat_model_name", "qwen3-max"))

        if provider in {"dashscope_compatible", "openai_compatible"}:
            try:
                from langchain_openai import ChatOpenAI

                base_url = (
                    model_cof.get("chat_base_url")
                    or os.getenv("DASHSCOPE_BASE_URL")
                    or os.getenv("DASH_SCOPE_BASE_URL")
                    or os.getenv("ZZZ_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
                print(api_key)
                return ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
            except Exception:
                return SimpleChatModel(model=model_name)

        if provider == "dashscope":
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi

                return ChatTongyi(model=model_name)
            except Exception:
                return SimpleChatModel(model=model_name)

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(model=model_name)
            except Exception:
                return SimpleChatModel(model=model_name)

        return SimpleChatModel(model=model_name)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Any:
        provider = model_cof.get("embedding_provider", rag_cof.get("embedding_provider", "dashscope"))
        model_name = model_cof.get("embedding_model_name", rag_cof.get("embedding_model_name", "text-embedding-v4"))

        if provider == "dashscope":
            try:
                from langchain_community.embeddings import DashScopeEmbeddings

                return DashScopeEmbeddings(model=model_name)
            except Exception:
                return SimpleEmbeddings()

        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings

                return OpenAIEmbeddings(model=model_name)
            except Exception:
                return SimpleEmbeddings()

        return SimpleEmbeddings()


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
