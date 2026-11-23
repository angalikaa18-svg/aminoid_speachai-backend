import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TTS_OUTPUT_DIR = BASE_DIR / "storage" / "tts_output"

class TTSConfig(BaseModel):
    engine_name: str = "sapi5_local"
    output_dir: str = str(TTS_OUTPUT_DIR)
    default_rate: int = 175          # Words per minute
    default_volume: float = 1.0      # 0.0 to 1.0
    default_pitch: int = 50          # 0 to 100
    format: str = "wav"

DEFAULT_TTS_CONFIG = TTSConfig()
