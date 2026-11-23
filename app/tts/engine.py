import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import DEFAULT_TTS_CONFIG, TTSConfig
from .text_preprocessor import TextPreprocessor
from .phonemizer import Phonemizer
from .voice_manager import VoiceManager
from .audio_generator import AudioGenerator
from .audio_postprocessor import AudioPostprocessor
from .schemas import TTSResponse, TTSVoice

class LocalTTSEngine:
    """100% offline Text-to-Speech synthesis pipeline."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or DEFAULT_TTS_CONFIG

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        pitch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes text into a local WAV file.
        Returns metadata including filename, duration, size, and relative audio URL.
        """
        # Step 1: Text cleaning and normalization
        prepared_text = TextPreprocessor.prepare(text)
        if not prepared_text:
            raise ValueError("Input text is empty after cleaning.")

        # Step 2: Phonemic & pause adjustment
        spoken_text = Phonemizer.insert_pause_markers(prepared_text)

        # Step 3: Determine output path
        unique_id = uuid.uuid4().hex[:12]
        filename = f"tts_{unique_id}.wav"
        output_path = os.path.join(self.config.output_dir, filename)

        # Step 4: Run local audio generation
        AudioGenerator.generate_wav(
            text=spoken_text,
            output_path=output_path,
            voice_id=voice_id,
            rate=rate,
            volume=volume,
            config=self.config
        )

        # Step 5: Post-process & measure file
        meta = AudioPostprocessor.process_file(output_path)

        voices = VoiceManager.get_available_voices()
        voice_name = next((v.name for v in voices if v.id == voice_id), "System Default")

        return {
            "success": True,
            "audio_url": f"/api/v1/tts/audio/{filename}",
            "filename": filename,
            "duration": meta["duration"],
            "file_size_bytes": meta["file_size_bytes"],
            "voice_used": voice_name,
            "text_preview": prepared_text[:80] + ("..." if len(prepared_text) > 80 else ""),
            "offline": True
        }

    @staticmethod
    def get_voices() -> List[TTSVoice]:
        return VoiceManager.get_available_voices()

    @staticmethod
    def get_status() -> Dict[str, Any]:
        return {
            "engine": DEFAULT_TTS_CONFIG.engine_name,
            "status": "ready",
            "voices_available": len(VoiceManager.get_available_voices()),
            "offline": True
        }
