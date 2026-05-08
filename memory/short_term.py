from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ShortTermMemory:
    window_size: int = 8
    messages: deque[dict] = field(default_factory=deque)
    summary: str = ""
    slots: dict = field(default_factory=dict)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        while len(self.messages) > self.window_size:
            self.messages.popleft()

    def set_slot(self, key: str, value) -> None:
        if value:
            self.slots[key] = value

    def snapshot(self) -> dict:
        return {
            "summary": self.summary,
            "messages": list(self.messages),
            "slots": self.slots.copy(),
        }
