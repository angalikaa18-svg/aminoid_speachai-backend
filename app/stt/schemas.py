from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0

class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = 1.0
    speaker: Optional[int] = None
    words: Optional[List[WordTimestamp]] = None

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: Optional[float] = 1.0
    duration: float
    segments: List[Segment]
    srt: str
    model: str = "faster-whisper-tiny (offline)"

class TranscribeOptions(BaseModel):
    language: Optional[str] = None
    beam_size: int = 1
    vad_filter: bool = True
    temperature: float = 0.0
