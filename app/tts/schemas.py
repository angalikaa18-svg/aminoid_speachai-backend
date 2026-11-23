from typing import Optional, List
from pydantic import BaseModel, Field

class TTSVoice(BaseModel):
    id: str
    name: str
    gender: str = "unknown"
    language: str = "en"
    age: Optional[str] = None
    engine: Optional[str] = "SAPI5"

class GenerateTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: Optional[str] = None
    rate: Optional[int] = 175
    volume: Optional[float] = 1.0
    pitch: Optional[int] = 50

class TTSResponse(BaseModel):
    success: bool
    audio_url: str
    filename: str
    duration: float
    file_size_bytes: int
    voice_used: str
    text_preview: str
    offline: bool = True
