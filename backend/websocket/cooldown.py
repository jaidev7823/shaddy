from typing import Dict
from backend.config import COOLDOWN

class CooldownManager:
    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
    
    def is_active(self, lesson_id: str) -> bool:
        if not lesson_id:
            return False
        import time
        now = time.time()
        return (now - self._cooldowns.get(lesson_id, 0)) < COOLDOWN
    
    def update(self, lesson_id: str):
        import time
        self._cooldowns[lesson_id] = time.time()

# Global instance shared across all connections
cooldown_manager = CooldownManager()
