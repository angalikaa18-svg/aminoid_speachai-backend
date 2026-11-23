import os
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from .config import DEFAULT_STT_CONFIG, STTConfig

class STTModelLoader:
    """Manages local model lifecycle with strict offline enforcement."""
    _instance: Optional[WhisperModel] = None
    _loaded_model_name: Optional[str] = None

    @classmethod
    def get_model(cls, config: Optional[STTConfig] = None) -> WhisperModel:
        cfg = config or DEFAULT_STT_CONFIG
        if cls._instance is not None and cls._loaded_model_name == cfg.model_name:
            return cls._instance

        model_dir = Path(cfg.models_dir) / cfg.model_name
        
        # Verify model directory exists locally
        if not model_dir.exists() or not (model_dir / "model.bin").exists():
            raise FileNotFoundError(
                f"Local STT model not found at '{model_dir}'. Run 'python scripts/setup_models.py' first."
            )

        # Enforce local_files_only=True so no internet requests can be triggered
        cls._instance = WhisperModel(
            str(model_dir),
            device=cfg.device,
            compute_type=cfg.compute_type,
            local_files_only=True,
            cpu_threads=2
        )
        cls._loaded_model_name = cfg.model_name
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None

    @classmethod
    def get_loaded_model_name(cls) -> str:
        return cls._loaded_model_name or "none"
