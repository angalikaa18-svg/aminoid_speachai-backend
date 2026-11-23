import os
import uuid
import pythoncom
import win32com.client
import pyttsx3
from pathlib import Path
from typing import Optional
from .config import DEFAULT_TTS_CONFIG, TTSConfig
from .voice_manager import VoiceManager

class AudioGenerator:
    """Invokes local SAPI5 & OneCore COM engines with thread-safe COM initialization."""

    @staticmethod
    def generate_wav(
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        config: Optional[TTSConfig] = None
    ) -> str:
        cfg = config or DEFAULT_TTS_CONFIG
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Thread-safe COM initialization for Windows background threads/FastAPI
        pythoncom.CoInitialize()
        try:
            target_voice = VoiceManager.find_voice(voice_id)
            actual_rate = rate if rate is not None else cfg.default_rate
            actual_volume = volume if volume is not None else cfg.default_volume
            normalized_volume = max(0.0, min(1.0, float(actual_volume)))

            # If voice belongs to Windows OneCore category, synthesize via SAPI.SpVoice with SpObjectTokenCategory
            if target_voice and "Speech_OneCore" in target_voice:
                sp = win32com.client.Dispatch("SAPI.SpVoice")
                cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
                cat.SetId(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", False)
                tokens = cat.EnumerateTokens()
                for i in range(tokens.Count):
                    if tokens.Item(i).Id == target_voice:
                        sp.Voice = tokens.Item(i)
                        break
                
                # SAPI Volume is 0 to 100
                sp.Volume = int(normalized_volume * 100)
                # Map standard WPM rate (100 - 250, default 175) to SAPI Rate (-10 to +10, 0 is default)
                sapi_rate = int((actual_rate - 175) / 15)
                sp.Rate = max(-10, min(10, sapi_rate))

                # Render to local WAV file stream
                fs = win32com.client.Dispatch("SAPI.SpFileStream")
                fs.Open(output_path, 3, False)
                sp.AudioOutputStream = fs
                sp.Speak(text)
                fs.Close()

                del fs
                del sp
                del tokens
                del cat
                return output_path
            else:
                # Use standard pyttsx3 engine for classic SAPI5 voices
                engine = pyttsx3.init()
                if target_voice:
                    engine.setProperty("voice", target_voice)

                engine.setProperty("rate", actual_rate)
                engine.setProperty("volume", normalized_volume)

                engine.save_to_file(text, output_path)
                engine.runAndWait()
                engine.stop()
                return output_path
        finally:
            pythoncom.CoUninitialize()
