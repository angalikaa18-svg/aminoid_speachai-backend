"""
AMINOID Speech AI — Python Integration Client SDK.
Designed for AI Receptionist, CallZenza, CRM, ERP, and third-party backend applications.
Communicates securely with AMINOID via API Keys (Bearer token authentication).
"""

import os
import json
import httpx
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

class AminoidClient:
    """
    Official Python Client for integrating external platforms with AMINOID Speech AI.
    
    Usage:
        client = AminoidClient(
            base_url=os.getenv("AMINOID_API_BASE_URL", "http://127.0.0.1:8080"),
            api_key=os.getenv("AMINOID_API_KEY", "aminoid_live_..."),
            client_id=os.getenv("AMINOID_CLIENT_ID", "callzenza_agent_01")
        )
        health = client.test_connection()
        transcription = client.transcribe(audio_path_or_bytes="caller.wav")
        speech = client.synthesize(text="Hello, thank you for calling CallZenza.")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = (base_url or os.getenv("AMINOID_API_BASE_URL") or "http://127.0.0.1:8080").rstrip("/")
        self.api_key = (api_key or os.getenv("AMINOID_API_KEY") or "").strip()
        self.client_id = client_id or os.getenv("AMINOID_CLIENT_ID") or "callzenza_receptionist"
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": f"AMINOID-Python-SDK/1.0 ({self.client_id})",
            "X-Client-ID": self.client_id
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def test_connection(self) -> Dict[str, Any]:
        """Tests connectivity and authentication with AMINOID backend."""
        url = f"{self.base_url}/api/v1/integrations/api-keys/test-connection"
        with httpx.Client(timeout=10.0) as client:
            res = client.post(url, json={"api_key": self.api_key}, headers=self._headers())
            res.raise_for_status()
            return res.json()

    def transcribe(
        self,
        audio: Union[str, Path, bytes],
        filename: str = "audio.wav",
        language: Optional[str] = None,
        vad_filter: bool = True,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio into text using AMINOID Local STT.
        Accepts audio file path (str/Path) or raw audio bytes.
        """
        url = f"{self.base_url}/api/v1/integrations/stt/transcribe"
        
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            filename = audio_path.name
            audio_bytes = audio_path.read_bytes()
        else:
            audio_bytes = audio

        files = {"audio": (filename, audio_bytes, "audio/wav")}
        data = {
            "language": language or "auto",
            "vad_filter": str(vad_filter).lower(),
            "word_timestamps": str(word_timestamps).lower()
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, files=files, data=data, headers=self._headers())
            res.raise_for_status()
            return res.json()

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        rate: int = 175,
        volume: float = 1.0,
        pitch: int = 50,
        response_mode: str = "json"
    ) -> Union[Dict[str, Any], bytes]:
        """
        Synthesizes speech from text using AMINOID Local TTS.
        If response_mode='json', returns metadata dict containing audio_url.
        If response_mode='audio', returns raw WAV audio bytes directly.
        """
        url = f"{self.base_url}/api/v1/integrations/tts/generate"
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
            "response_mode": response_mode
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload, headers=self._headers())
            res.raise_for_status()
            if response_mode == "audio":
                return res.content
            return res.json()

    def list_voices(self) -> List[Dict[str, Any]]:
        """Retrieves available voice acoustic profiles."""
        url = f"{self.base_url}/api/v1/integrations/tts/voices"
        with httpx.Client(timeout=10.0) as client:
            res = client.get(url, headers=self._headers())
            res.raise_for_status()
            return res.json().get("voices", [])

    def list_models(self) -> Dict[str, Any]:
        """Retrieves active models diagnostic status."""
        url = f"{self.base_url}/api/v1/integrations/models"
        with httpx.Client(timeout=10.0) as client:
            res = client.get(url, headers=self._headers())
            res.raise_for_status()
            return res.json()

    def get_voice_library(self) -> Dict[str, Any]:
        """Retrieves custom voice profiles and personas."""
        url = f"{self.base_url}/api/v1/integrations/voice-library"
        with httpx.Client(timeout=10.0) as client:
            res = client.get(url, headers=self._headers())
            res.raise_for_status()
            return res.json()
