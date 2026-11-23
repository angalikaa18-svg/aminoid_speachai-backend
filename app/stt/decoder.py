from typing import Any, Tuple, Optional
import numpy as np
from .model_loader import STTModelLoader
from .config import DEFAULT_STT_CONFIG, STTConfig

class STTDecoder:
    """Invokes local Whisper CTranslate2 engine for inference."""

    @staticmethod
    def decode(
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        config: Optional[STTConfig] = None
    ) -> Tuple[list, Any]:
        model = STTModelLoader.get_model(config)
        lang_arg = None if (not language or language.lower() in ("auto", "none")) else language.lower()

        segments_gen, info = model.transcribe(
            audio,
            language=lang_arg,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            temperature=0.0
        )
        # Materialize generator
        segments = list(segments_gen)
        return segments, info
