from pathlib import Path
from typing import Dict, Any
from .config import DEFAULT_TTS_CONFIG

class TTSModelLoader:
    """Tracks local TTS engine state and hardware configuration."""

    @staticmethod
    def get_status() -> Dict[str, Any]:
        return {
            "engine": DEFAULT_TTS_CONFIG.engine_name,
            "status": "ready",
            "offline": True,
            "hardware": "cpu"
        }
