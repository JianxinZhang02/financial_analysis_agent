from __future__ import annotations

import re

from utils.config_handler import memory_cof


class MemoryHeartbeat:
    def __init__(self):
        self.message_threshold = int(memory_cof.get("heartbeat_message_threshold", 12))
        self.char_threshold = int(memory_cof.get("heartbeat_char_threshold", 16000))

    def should_compact(self, messages: list[dict]) -> bool:
        total_chars = sum(len(message.get("content", "")) for message in messages)
        return len(messages) >= self.message_threshold or total_chars >= self.char_threshold

    def summarize(self, messages: list[dict], previous_summary: str = "") -> str:
        text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        companies = sorted(set(re.findall(r"[\u4e00-\u9fffA-Za-z]+(?:科技|股份|银行|证券|集团|公司)", text)))
        metrics = sorted(
            set(
                metric
                for metric in ["营收", "净利润", "毛利率", "现金流", "ROE", "市盈率", "资产负债率"]
                if metric in text
            )
        )
        parts = []
        if previous_summary:
            parts.append(previous_summary)
        if companies:
            parts.append("关注实体：" + "、".join(companies[:8]))
        if metrics:
            parts.append("关注指标：" + "、".join(metrics))
        parts.append("最近对话摘要：" + text[-1200:])
        return "\n".join(parts)
