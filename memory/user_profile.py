from __future__ import annotations

from memory.long_term import LongTermMemory


class UserProfileService:
    def __init__(self):
        self.memory = LongTermMemory()

    def get(self, user_id: str = "default") -> dict:
        return self.memory.get_profile(user_id)

    def remember_focus(self, user_id: str, companies: list[str], metrics: list[str]) -> dict:
        updates = {}
        if companies:
            updates["watchlist"] = companies
        if metrics:
            updates["preferred_metrics"] = metrics
        return self.memory.update_profile(user_id, **updates)
