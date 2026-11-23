from typing import Dict, Any
from pathlib import Path

class Vocoder:
    """Acoustic waveform representation layer for local speech generation."""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "native_sapi5_vocoder",
            "channels": 1,
            "sample_rate": 22050
        }
