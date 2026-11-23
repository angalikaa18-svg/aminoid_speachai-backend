import os
import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.stt.engine import LocalSTTEngine
from app.tts.engine import LocalTTSEngine
from app.audio.processor import AudioProcessor

client = TestClient(app)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE_WAV = ROOT_DIR / "examples" / "fixtures" / "audio.wav"

# -------------------------------------------------------------
# 1. Health & Status Regression Tests
# -------------------------------------------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "100% Offline"
    assert data["cloud_providers"] == "None (Self-Hosted)"
    assert "stt" in data and "tts" in data

def test_index_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "Local Speech Platform" in response.text
    assert "OFFLINE VERIFIED" in response.text

# -------------------------------------------------------------
# 2. Local STT Engine & /transcribe Endpoint Tests
# -------------------------------------------------------------
def test_transcribe_endpoint_valid_audio():
    assert SAMPLE_WAV.exists(), "Sample WAV fixture not found"
    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"language": "en"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert len(data["text"]) > 0
    assert data["language"] == "en"
    assert "segments" in data and len(data["segments"]) > 0
    assert "srt" in data and "-->" in data["srt"]
    assert data["offline"] is True
    assert "model" in data

def test_transcribe_empty_audio():
    empty_bytes = io.BytesIO(b"")
    response = client.post(
        "/transcribe",
        files={"file": ("empty.wav", empty_bytes, "audio/wav")}
    )
    assert response.status_code == 400

def test_transcribe_unsupported_format():
    fake_file = io.BytesIO(b"some invalid text data")
    response = client.post(
        "/transcribe",
        files={"file": ("test.xyz", fake_file, "text/plain")}
    )
    assert response.status_code == 400

def test_transcribe_url_blocked_for_privacy():
    response = client.post("/transcribe/url")
    assert response.status_code == 403

# -------------------------------------------------------------
# 3. Local TTS Engine & /api/v1/tts/ Endpoints Tests
# -------------------------------------------------------------
def test_tts_voices_list():
    response = client.get("/api/v1/tts/voices")
    assert response.status_code == 200
    voices = response.json()
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert "id" in voices[0]
    assert "name" in voices[0]

def test_tts_generate_endpoint():
    payload = {
        "text": "This is an automated test of the offline text to speech engine.",
        "rate": 180,
        "volume": 1.0
    }
    response = client.post("/api/v1/tts/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "filename" in data
    assert data["duration"] > 0
    assert data["offline"] is True

    # Test downloading the generated audio
    audio_resp = client.get(data["audio_url"])
    assert audio_resp.status_code == 200
    assert len(audio_resp.content) > 0
    assert audio_resp.headers["content-type"] == "audio/wav"

# -------------------------------------------------------------
# 4. Model Status & Management Tests
# -------------------------------------------------------------
def test_models_status():
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["offline_verified"] is True
    assert data["stt"]["model_name"] == "faster-whisper-tiny"

# -------------------------------------------------------------
# 5. Strict Privacy & Network Isolation Checks
# -------------------------------------------------------------
def test_zero_deepgram_in_app():
    app_dir = ROOT_DIR / "app"
    for py_file in app_dir.rglob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            assert "deepgramclient" not in content, f"DeepgramClient found in {py_file}"
            assert "deepgram.com" not in content, f"External Deepgram URL found in {py_file}"
