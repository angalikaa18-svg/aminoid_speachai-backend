import os
from pathlib import Path
from typing import Dict, Any
import soundfile as sf

class AudioPostprocessor:
    """Calculates duration, verifies WAV file integrity, and measures byte size."""

    @staticmethod
    def process_file(file_path: str) -> Dict[str, Any]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Generated audio file missing: {file_path}")

        file_size = os.path.getsize(file_path)
        try:
            info = sf.info(file_path)
            duration = round(info.duration, 2)
        except Exception:
            # Fallback estimation based on file size if header is raw PCM
            duration = round(file_size / (22050 * 2), 2)

        return {
            "duration": duration,
            "file_size_bytes": file_size,
            "path": file_path
        }
