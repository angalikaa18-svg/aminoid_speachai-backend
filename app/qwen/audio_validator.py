"""
Audio Preprocessing and Quality Validation for Qwen3-TTS Voice Cloning.
Follows strict acoustic conditioning guidelines:
- Mono float32 conversion
- 24 kHz resampling for neural speaker encoder
- Controlled peak normalization to -1.0 dBFS
- Dead silence trimming (without altering speech prosody or pauses)
- Audio health checks (energy level, clipping detection, duration bounds)
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 24000  # Qwen3-TTS speaker encoder sample rate
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 60.0
RECOMMENDED_MIN_SEC = 3.0
RMS_SILENCE_THRESHOLD = 0.005  # Below this is considered essentially silent
CLIPPING_THRESHOLD = 0.995

def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validates audio file quality, duration, and signal integrity.
    Returns detailed diagnostics.
    """
    if not os.path.exists(file_path):
        return {
            "valid": False,
            "error": "Reference audio file does not exist."
        }

    try:
        # Load audio using soundfile to inspect raw file properties
        info = sf.info(file_path)
        original_sr = info.samplerate
        channels = info.channels
        raw_duration = info.duration

        if raw_duration < MIN_DURATION_SEC:
            return {
                "valid": False,
                "error": f"Reference audio too short ({raw_duration:.1f}s). Minimum required is {MIN_DURATION_SEC}s.",
                "duration": raw_duration
            }

        if raw_duration > MAX_DURATION_SEC:
            return {
                "valid": False,
                "error": f"Reference audio exceeds maximum duration of {MAX_DURATION_SEC}s ({raw_duration:.1f}s).",
                "duration": raw_duration
            }

        # Load samples for acoustic inspection
        audio_data, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_mono = np.mean(audio_data, axis=-1)
        else:
            audio_mono = audio_data

        # RMS Energy check
        rms = float(np.sqrt(np.mean(audio_mono ** 2)))
        if rms < RMS_SILENCE_THRESHOLD:
            return {
                "valid": False,
                "error": "Reference audio contains insufficient speech or is too quiet. Please provide a clearer recording.",
                "duration": raw_duration,
                "rms": rms
            }

        # Peak clipping check
        peak = float(np.abs(audio_mono).max())
        is_clipping = peak >= CLIPPING_THRESHOLD
        warning = None
        if is_clipping:
            warning = "Reference audio appears to be clipping near maximum level. A lower microphone gain is recommended."
        elif raw_duration < RECOMMENDED_MIN_SEC:
            warning = f"Reference recording is {raw_duration:.1f}s. Recordings between 4s and 15s provide optimal cloning fidelity."

        return {
            "valid": True,
            "original_sr": original_sr,
            "channels": channels,
            "duration": round(raw_duration, 2),
            "rms_energy": round(rms, 4),
            "peak_amplitude": round(peak, 4),
            "is_clipping": is_clipping,
            "warning": warning
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Audio inspection error: {str(e)}"
        }

def preprocess_reference_audio(input_path: str, output_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocesses reference audio strictly for Qwen3-TTS conditioning:
    1. Converts stereo to mono
    2. Resamples to 24,000 Hz using high-quality sinc/polyphase filter
    3. Trims leading and trailing silence (top_db=28) preserving speech
    4. Peak-normalizes to -1.0 dBFS (0.891 factor) to eliminate distortion
    Preserves original file untouched.
    """
    validation = validate_audio_file(input_path)
    if not validation.get("valid"):
        raise ValueError(validation.get("error", "Invalid reference audio."))

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"proc_{p.stem}.wav")

    # Load audio as mono
    y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

    # Trim leading and trailing silence only (preserve interior pauses for natural speech cadence)
    y_trimmed, trim_idx = librosa.effects.trim(y, top_db=28, frame_length=512, hop_length=128)
    trimmed_duration = float(len(y_trimmed) / TARGET_SR)

    if trimmed_duration < MIN_DURATION_SEC:
        # If trimming was too aggressive, retain original resampled audio
        y_trimmed = y
        trimmed_duration = float(len(y) / TARGET_SR)

    # Peak normalization to -1.0 dBFS (prevents digital clipping during codec quantization)
    max_val = np.abs(y_trimmed).max()
    if max_val > 0:
        target_peak = 10 ** (-1.0 / 20.0)  # ~0.89125
        y_norm = (y_trimmed / max_val) * target_peak
    else:
        y_norm = y_trimmed

    # Write clean 24kHz 16-bit PCM WAV
    sf.write(output_path, y_norm.astype(np.float32), TARGET_SR, subtype="PCM_16")

    meta = {
        "processed_path": output_path,
        "sample_rate": TARGET_SR,
        "channels": 1,
        "duration": round(trimmed_duration, 2),
        "validation": validation
    }
    return output_path, meta
