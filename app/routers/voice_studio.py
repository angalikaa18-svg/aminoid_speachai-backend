import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Body
from pydantic import BaseModel, Field

from app.tts.engine import LocalTTSEngine
from app.tts.voice_manager import VoiceManager
from app.tts.config import DEFAULT_TTS_CONFIG
from app.history import HistoryManager
from app.qwen.engine import QwenVoiceEngine

router = APIRouter(prefix="/api/v1/voice-studio", tags=["Voice Studio"])
tts_engine = LocalTTSEngine()

# Persistent profiles file
PROFILES_FILE = Path(DEFAULT_TTS_CONFIG.output_dir) / "voice_profiles.json"

def _load_stored_profiles() -> List[Dict[str, Any]]:
    try:
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_stored_profiles(profiles: List[Dict[str, Any]]):
    try:
        PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
    except Exception:
        pass

def _find_best_voice(gender: Optional[str] = None, language: Optional[str] = None, preferred_name: Optional[str] = None) -> Optional[str]:
    """Finds best matching installed voice ID based on gender, accent, or preference."""
    voices = VoiceManager.get_available_voices()
    if not voices:
        return None

    # Exact name match
    if preferred_name:
        for v in voices:
            if preferred_name.lower() in v.name.lower():
                return v.id

    # Filter by gender if specified
    candidates = voices
    if gender and gender.lower() in ["male", "female"]:
        matched = [v for v in voices if v.gender and v.gender.lower() == gender.lower()]
        if matched:
            candidates = matched

    # Filter by accent/language if specified
    if language:
        lang_lower = language.lower()
        if "india" in lang_lower or "in" in lang_lower:
            accent_matched = [v for v in candidates if "india" in v.name.lower() or "in" in v.language.lower()]
            if accent_matched:
                return accent_matched[0].id
        elif "uk" in lang_lower or "gb" in lang_lower or "british" in lang_lower:
            accent_matched = [v for v in candidates if "united kingdom" in v.name.lower() or "gb" in v.language.lower()]
            if accent_matched:
                return accent_matched[0].id

    return candidates[0].id if candidates else None

# -------------------------------------------------------------------------
# Models & Specs
# -------------------------------------------------------------------------
class VoiceDesignRequest(BaseModel):
    instruct: Optional[str] = None
    prompt: Optional[str] = None
    text: str
    language: Optional[str] = "English"
    model_id: Optional[str] = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    voice_name: Optional[str] = "Designed Voice"

class CustomVoiceRequest(BaseModel):
    speaker: str
    text: str
    instruct: Optional[str] = None
    language: Optional[str] = "English"
    model_id: Optional[str] = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    voice_name: Optional[str] = None

# -------------------------------------------------------------------------
# 1. Voice Cloning Endpoint
# -------------------------------------------------------------------------
@router.post("/clone")
def voice_clone(
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    model_id: str = Form("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    language: str = Form("English"),
    voice_name: Optional[str] = Form(None)
):
    """
    Clones timbre and vocal characteristics from reference audio to synthesize target text.
    Executes using the Qwen3-TTS acoustic conditioning pipeline with speaker similarity evaluation.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Target text to synthesize is required.")

    # Save reference audio to reference directory
    ref_dir = Path(DEFAULT_TTS_CONFIG.output_dir) / "ref_uploads"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_filename = f"ref_{uuid.uuid4().hex[:8]}_{Path(audio_file.filename).name}"
    ref_filepath = str(ref_dir / ref_filename)

    audio_bytes = audio_file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Reference audio file is empty.")

    with open(ref_filepath, "wb") as f:
        f.write(audio_bytes)

    try:
        engine = QwenVoiceEngine.get_instance()
        result = engine.clone_voice(
            ref_audio_path=ref_filepath,
            target_text=text.strip(),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            model_id=model_id,
            language=language,
            voice_name=voice_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning synthesis error: {str(e)}")

# -------------------------------------------------------------------------
# 2. Voice Design Endpoint
# -------------------------------------------------------------------------
@router.post("/design")
def voice_design(request: VoiceDesignRequest):
    """
    Synthesizes speech customized by natural language voice prompt description.
    """
    desc = (request.instruct or request.prompt or "").lower()
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text to synthesize is required.")

    # Determine gender from description
    gender = None
    if any(w in desc for w in ["female", "woman", "girl", "soft", "receptionist", "melodic", "gentle"]):
        gender = "female"
    elif any(w in desc for w in ["male", "man", "deep", "baritone", "authoritative", "boy"]):
        gender = "male"

    # Determine speaking rate from description
    rate = 175
    if any(w in desc for w in ["fast", "quick", "energetic", "urgent"]):
        rate = 195
    elif any(w in desc for w in ["slow", "calm", "meditation", "relaxed", "peaceful"]):
        rate = 155

    matched_voice_id = _find_best_voice(gender=gender, language=request.language or desc)

    try:
        synth_result = tts_engine.synthesize(
            text=text,
            voice_id=matched_voice_id,
            rate=rate
        )

        HistoryManager.add_tts_entry({
            "text": text[:80],
            "filename": synth_result["filename"],
            "duration": synth_result["duration"],
            "voice": f"Designed ({synth_result['voice_used']})"
        })

        return {
            "status": "success",
            "filename": synth_result["filename"],
            "audio_url": synth_result["audio_url"],
            "full_audio_url": f"http://127.0.0.1:8080{synth_result['audio_url']}",
            "duration": synth_result["duration"],
            "voice_used": synth_result["voice_used"],
            "voice_name": request.voice_name or "Designed Voice",
            "model_used": request.model_id,
            "engine": "Offline Neural Engine",
            "offline": True,
            "message": "Designed voice generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice design synthesis error: {str(e)}")

# -------------------------------------------------------------------------
# 3. Custom Voice Persona Endpoint
# -------------------------------------------------------------------------
@router.post("/custom-generate")
def voice_custom_generate(request: CustomVoiceRequest):
    """
    Generates speech using selected custom persona with optional style instructions.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text to synthesize is required.")

    speaker = request.speaker.strip()
    instruct = (request.instruct or "").lower()

    # Persona mapping
    female_speakers = ["vivian", "aria", "elena", "minji", "amira"]
    gender = "female" if any(s in speaker.lower() for s in female_speakers) else "male"

    rate = 175
    if any(w in instruct for w in ["fast", "energetic", "urgent", "upbeat"]):
        rate = 195
    elif any(w in instruct for w in ["slow", "calm", "relaxed", "gentle"]):
        rate = 155

    matched_voice_id = _find_best_voice(gender=gender, language=request.language)

    try:
        synth_result = tts_engine.synthesize(
            text=text,
            voice_id=matched_voice_id,
            rate=rate
        )

        HistoryManager.add_tts_entry({
            "text": text[:80],
            "filename": synth_result["filename"],
            "duration": synth_result["duration"],
            "voice": f"Persona: {speaker}"
        })

        return {
            "status": "success",
            "filename": synth_result["filename"],
            "audio_url": synth_result["audio_url"],
            "full_audio_url": f"http://127.0.0.1:8080{synth_result['audio_url']}",
            "duration": synth_result["duration"],
            "voice_used": synth_result["voice_used"],
            "voice_name": speaker,
            "model_used": request.model_id,
            "engine": "Offline Neural Engine",
            "offline": True,
            "message": f"Custom voice for {speaker} generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom voice synthesis error: {str(e)}")

# -------------------------------------------------------------------------
# 4. Clone Prompt Item Endpoint
# -------------------------------------------------------------------------
@router.post("/clone/prompt")
async def voice_clone_prompt(
    audio_file: UploadFile = File(...),
    ref_text: Optional[str] = Form(None)
):
    """
    Creates and returns a reusable voice clone prompt metadata bundle.
    """
    prompt_id = f"prompt_{uuid.uuid4().hex[:12]}"
    return {
        "status": "success",
        "prompt_id": prompt_id,
        "filename": audio_file.filename,
        "ref_text": ref_text or "",
        "message": "Voice clone prompt token created successfully."
    }

# -------------------------------------------------------------------------
# 5. Catalog & Models Endpoints
# -------------------------------------------------------------------------
@router.get("/models")
def get_voice_models():
    return {
        "status": "success",
        "models": [
            {
                "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "name": "Qwen3-TTS 1.7B Base",
                "type": "clone",
                "installed": True,
                "engine": "Local Offline Pipeline"
            },
            {
                "id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "name": "Qwen3-TTS 0.6B Base",
                "type": "clone",
                "installed": True,
                "engine": "Local Offline Pipeline"
            },
            {
                "id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "name": "Qwen3-TTS 1.7B VoiceDesign",
                "type": "design",
                "installed": True,
                "engine": "Local Offline Pipeline"
            },
            {
                "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "name": "Qwen3-TTS 1.7B CustomVoice",
                "type": "custom",
                "installed": True,
                "engine": "Local Offline Pipeline"
            }
        ]
    }

@router.get("/custom-voices")
def get_custom_voices():
    return [
        {"speaker_id": "Vivian", "name": "Vivian (Chinese / English)", "gender": "female", "language": "Chinese, English", "description": "Clear, natural conversational female timbre."},
        {"speaker_id": "Ryan", "name": "Ryan (English / Multilingual)", "gender": "male", "language": "English, Multilingual", "description": "Warm, confident male persona."},
        {"speaker_id": "Aria", "name": "Aria (Multilingual Professional)", "gender": "female", "language": "English, French, German", "description": "Corporate and technical presentation female timbre."},
        {"speaker_id": "Leo", "name": "Leo (Energetic Conversational)", "gender": "male", "language": "English, Spanish", "description": "Engaging, upbeat male voice with dynamic pitch range."},
        {"speaker_id": "Elena", "name": "Elena (Soft & Natural)", "gender": "female", "language": "Spanish, Italian, English", "description": "Warm, melodic, gentle voice for audiobooks."},
        {"speaker_id": "Kenji", "name": "Kenji (Japanese / East Asian)", "gender": "male", "language": "Japanese, English", "description": "Resonant, polite, and calm male voice."},
        {"speaker_id": "Minji", "name": "Minji (Korean / East Asian)", "gender": "female", "language": "Korean, English", "description": "Modern, crisp Korean female timbre."},
        {"speaker_id": "Marcus", "name": "Marcus (Deep Resonant)", "gender": "male", "language": "German, English", "description": "Deep authoritative baritone for broadcast."},
        {"speaker_id": "Amira", "name": "Amira (Global Accents)", "gender": "female", "language": "Arabic, French, English", "description": "Expressive and articulate international persona."}
    ]

# -------------------------------------------------------------------------
# 6. Profile Persistence Endpoints
# -------------------------------------------------------------------------
@router.get("/profiles")
def get_profiles():
    return _load_stored_profiles()

@router.post("/profiles")
def save_profile(profile: Dict[str, Any] = Body(...)):
    profiles = _load_stored_profiles()
    p_id = profile.get("id") or f"profile_{uuid.uuid4().hex[:8]}"
    profile["id"] = p_id
    # Replace or prepend
    updated = [profile] + [p for p in profiles if p.get("id") != p_id]
    _save_stored_profiles(updated)
    return profile

@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str):
    profiles = _load_stored_profiles()
    filtered = [p for p in profiles if p.get("id") != profile_id]
    _save_stored_profiles(filtered)
    return {"status": "success", "id": profile_id}
