from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config_handler import memory_cof
from utils.path_tool import get_abs_path


class LongTermMemory:
    def __init__(self, path: str | None = None):
        self.path = Path(get_abs_path(path or memory_cof["profile_store_path"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self._data.setdefault(
            user_id,
            {
                "watchlist": [],
                "preferred_metrics": [],
                "risk_preference": "unknown",
                "language_style": "professional",
                "history_topics": [],
            },
        )

    def update_profile(self, user_id: str, **kwargs) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        for key, value in kwargs.items():
            if isinstance(profile.get(key), list):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if item and item not in profile[key]:
                        profile[key].append(item)
            elif value:
                profile[key] = value
        self.save()
        return profile
