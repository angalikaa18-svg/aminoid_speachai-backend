import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac", ".wma"}
DEFAULT_SAMPLE_RATE = 16000

def find_ffmpeg() -> str:
    """Locate ffmpeg executable in PATH or WinGet packages."""
    cmd = shutil.which("ffmpeg")
    if cmd:
        return cmd
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    winget_pkg_dir = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_pkg_dir):
        for entry in os.listdir(winget_pkg_dir):
            if "FFmpeg" in entry:
                for root, dirs, files in os.walk(os.path.join(winget_pkg_dir, entry)):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")
    return "ffmpeg"

FFMPEG_PATH = find_ffmpeg()

class AudioProcessor:
    """100% offline audio decoding, resampling, and preprocessing pipeline."""

    @staticmethod
    def validate_format(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    @staticmethod
    def save_temp_audio(data: bytes, suffix: str = ".wav") -> str:
        """Securely persist in-memory audio bytes to a temporary local file."""
        temp_dir = tempfile.gettempdir()
        unique_name = f"local_speech_{uuid.uuid4().hex[:12]}{suffix}"
        temp_path = os.path.join(temp_dir, unique_name)
        with open(temp_path, "wb") as f:
            f.write(data)
        return temp_path

    @staticmethod
    def cleanup_temp_file(path: Optional[str]) -> None:
        """Safely remove a temporary audio file."""
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass

    @classmethod
    def decode_and_resample(
        cls, 
        audio_input: Any, 
        target_sr: int = DEFAULT_SAMPLE_RATE
    ) -> Tuple[np.ndarray, float]:
        """
        Convert any supported audio format into a 16kHz mono float32 numpy array.
        Returns:
            (audio_array: np.ndarray, duration_seconds: float)
        """
        temp_file = None
        if isinstance(audio_input, (str, Path)):
            input_path = str(audio_input)
        elif isinstance(audio_input, bytes):
            temp_file = cls.save_temp_audio(audio_input, ".tmp")
            input_path = temp_file
        else:
            raise ValueError("audio_input must be a file path string or raw bytes.")

        try:
            # First try direct ffmpeg stream decoding to raw PCM 16-bit mono 16kHz
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i", input_path,
                "-vn",               # Disable video stream
                "-ar", str(target_sr),
                "-ac", "1",          # Mono
                "-f", "s16le",       # 16-bit signed PCM
                "-"
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            raw_pcm, stderr = proc.communicate()

            if proc.returncode == 0 and raw_pcm:
                audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                duration = len(audio_np) / float(target_sr)
                return audio_np, duration

            # Fallback: soundfile for standard WAV/FLAC/OGG
            try:
                data, sr = sf.read(input_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1) # Convert stereo to mono
                if sr != target_sr:
                    # Simple linear or step resampling if ffmpeg wasn't used
                    indices = np.round(np.arange(0, len(data), sr / target_sr)).astype(int)
                    indices = indices[indices < len(data)]
                    data = data[indices]
                data = data.astype(np.float32)
                duration = len(data) / float(target_sr)
                return data, duration
            except Exception as sf_err:
                raise RuntimeError(
                    f"FFmpeg and SoundFile both failed to decode audio: {stderr.decode(errors='ignore') or sf_err}"
                )

        finally:
            if temp_file:
                cls.cleanup_temp_file(temp_file)

    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """Retrieve local audio file duration, sample rate, and format metadata."""
        try:
            info = sf.info(file_path)
            return {
                "samplerate": info.samplerate,
                "channels": info.channels,
                "duration": round(info.duration, 2),
                "format": info.format
            }
        except Exception:
            return {
                "samplerate": DEFAULT_SAMPLE_RATE,
                "channels": 1,
                "duration": 0.0,
                "format": Path(file_path).suffix.lstrip(".").upper()
            }
