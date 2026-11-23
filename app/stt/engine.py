from typing import Any, Optional, Dict
from .audio_preprocessor import STTAudioPreprocessor
from .decoder import STTDecoder
from .segmenter import Segmenter
from .timestamp_processor import TimestampProcessor
from .normalizer import TextNormalizer
from .language_detector import LanguageDetector
from .config import DEFAULT_STT_CONFIG, STTConfig
from .model_loader import STTModelLoader

class LocalSTTEngine:
    """Complete 100% offline Speech-to-Text inference pipeline."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or DEFAULT_STT_CONFIG

    def transcribe(
        self,
        audio_input: Any,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: Optional[bool] = None,
        word_timestamps: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Transcribes audio completely on local hardware.
        Accepts raw bytes, file path string, or Path object.
        Returns normalized response dictionary matching existing API contract.
        """
        # Step 1: Decode and resample audio locally
        audio_array, duration = STTAudioPreprocessor.process(audio_input)
        if len(audio_array) == 0:
            return {
                "text": "",
                "language": "unknown",
                "segments": [],
                "srt": "",
                "duration": 0.0,
                "model": self.config.model_name
            }

        # Step 2: Run local model inference
        raw_segments, info = STTDecoder.decode(
            audio=audio_array,
            language=language or self.config.default_language,
            beam_size=beam_size if beam_size is not None else self.config.beam_size,
            vad_filter=vad_filter if vad_filter is not None else self.config.vad_filter,
            word_timestamps=word_timestamps if word_timestamps is not None else self.config.word_timestamps,
            config=self.config
        )

        # Step 3: Segment and timestamp processing
        segments = Segmenter.process_segments(raw_segments)
        full_text = " ".join(seg["text"] for seg in segments)
        normalized_text = TextNormalizer.normalize(full_text)

        # Step 4: SRT caption construction
        srt_content = TimestampProcessor.generate_srt(segments)

        detected_lang = getattr(info, "language", "unknown")

        return {
            "text": normalized_text,
            "language": detected_lang,
            "language_name": LanguageDetector.get_language_name(detected_lang),
            "language_probability": round(getattr(info, "language_probability", 1.0), 3),
            "duration": round(duration, 2),
            "segments": segments,
            "srt": srt_content,
            "model": self.config.model_name,
            "device": self.config.device,
            "offline": True
        }

    @staticmethod
    def get_status() -> Dict[str, Any]:
        return {
            "loaded": STTModelLoader.is_loaded(),
            "model_name": STTModelLoader.get_loaded_model_name(),
            "device": DEFAULT_STT_CONFIG.device,
            "offline": True
        }
