import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "stt"
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"

class STTConfig(BaseModel):
    model_name: str = "faster-whisper-tiny"
    models_dir: str = str(MODELS_DIR)
    upload_dir: str = str(UPLOAD_DIR)
    device: str = "cpu"
    compute_type: str = "int8"
    local_files_only: bool = True
    beam_size: int = 1
    vad_filter: bool = True
    default_language: str = "auto"
    word_timestamps: bool = True

DEFAULT_STT_CONFIG = STTConfig()
