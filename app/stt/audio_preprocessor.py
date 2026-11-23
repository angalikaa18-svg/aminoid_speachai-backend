from typing import Any, Tuple
import numpy as np
from ..audio.processor import AudioProcessor

class STTAudioPreprocessor:
    """Prepares and validates audio arrays for the local Whisper decoder."""

    @staticmethod
    def process(audio_input: Any) -> Tuple[np.ndarray, float]:
        """Convert input file path or bytes to 16kHz float32 mono array."""
        return AudioProcessor.decode_and_resample(audio_input, target_sr=16000)
