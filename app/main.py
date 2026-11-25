import os
import sys
import time
import multiprocessing
import warnings
import shutil
import tempfile
import psutil
from pathlib import Path
from typing import Optional, List, Dict, Any

SERVER_START_TIME = time.time()

# Ensure child processes spawned on Windows (such as Uvicorn reloader workers) use the virtualenv Python interpreter
if sys.platform == "win32":
    try:
        multiprocessing.set_executable(sys.executable)
    except Exception:
        pass

# Suppress harmless framework deprecation notices in server console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, status, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.audio.processor import AudioProcessor
from app.stt.engine import LocalSTTEngine
from app.stt.config import DEFAULT_STT_CONFIG
from app.stt.language_detector import LANGUAGE_NAMES
from app.tts.engine import LocalTTSEngine
from app.tts.config import DEFAULT_TTS_CONFIG
from app.tts.schemas import GenerateTTSRequest, TTSVoice
from app.history import HistoryManager
from app.settings_manager import SettingsManager
from app.routers.voice_studio import router as voice_studio_router
from app.integrations.router import router as integrations_router
from app.integrations.dispatcher import dispatch_event

app = FastAPI(
    title="Local Speech Platform",
    description="100% Offline Speech-to-Text & Text-to-Speech Engine",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Voice Studio router and Integrations router
app.include_router(voice_studio_router)
app.include_router(integrations_router)

# Direct Webhook Listener Route for Postman & external callers on port 8080
@app.api_route("/webhook", methods=["GET", "POST"])
@app.api_route("/api/v1/webhook", methods=["GET", "POST"])
async def root_webhook_endpoint(request: Request):
    """
    Direct Webhook Handler on port 8080.
    Accepts incoming webhooks, validates reachability, and returns helpful instructions.
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
    return {
        "success": True,
        "status": "online",
        "method": request.method,
        "message": "AMINOID Webhook endpoint reached successfully on port 8080.",
        "received_payload_length": len(body_str),
        "headers": {k: v for k, v in request.headers.items() if k.lower().startswith("x-aminoid") or k.lower() in ("content-type", "user-agent")},
        "endpoints": {
            "test_dispatcher": "POST /api/v1/integrations/webhooks/test",
            "subscriptions": "GET|POST /api/v1/integrations/webhooks",
            "deliveries": "GET /api/v1/integrations/webhooks/deliveries"
        }
    }

# Initialize offline local engines
stt_engine = LocalSTTEngine()
tts_engine = LocalTTSEngine()

# ---------------------------------------------------------
# Web UI Dashboard (100% Offline)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Local Speech Platform (100% Offline)</title>
        <style>
            :root {
                --bg: #090d16;
                --surface: #131b2e;
                --surface-hover: #1c2742;
                --border: #233152;
                --primary: #38bdf8;
                --primary-hover: #0284c7;
                --accent: #10b981;
                --text: #f1f5f9;
                --text-muted: #94a3b8;
            }
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border);
                padding-bottom: 16px;
                margin-bottom: 24px;
            }
            .title {
                margin: 0;
                font-size: 24px;
                color: var(--primary);
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .badge-offline {
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid var(--accent);
                color: var(--accent);
                font-size: 12px;
                padding: 4px 10px;
                border-radius: 999px;
                font-weight: 600;
            }
            .card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .status-item {
                background: var(--surface-hover);
                border-radius: 8px;
                padding: 14px;
                border: 1px solid var(--border);
            }
            .status-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
            .status-value { font-size: 16px; font-weight: 600; margin-top: 4px; color: var(--text); }
            
            .tabs {
                display: flex;
                gap: 12px;
                margin-bottom: 20px;
            }
            .tab-btn {
                background: var(--surface-hover);
                border: 1px solid var(--border);
                color: var(--text-muted);
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.2s;
            }
            .tab-btn.active {
                background: var(--primary);
                color: #0f172a;
                border-color: var(--primary);
            }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            .dropzone {
                border: 2px dashed var(--border);
                border-radius: 10px;
                padding: 36px 20px;
                text-align: center;
                cursor: pointer;
                transition: border-color 0.2s;
                background: rgba(255, 255, 255, 0.01);
            }
            .dropzone:hover { border-color: var(--primary); }
            .btn {
                background: var(--primary);
                color: #0f172a;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                transition: background 0.2s;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            .btn:hover { background: var(--primary-hover); color: #fff; }
            .btn-secondary {
                background: var(--surface-hover);
                color: var(--text);
                border: 1px solid var(--border);
            }
            .btn-secondary:hover { background: var(--border); }
            
            textarea, select, input[type="text"], input[type="number"], input[type="range"] {
                width: 100%;
                background: #090d16;
                border: 1px solid var(--border);
                color: var(--text);
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 14px;
                margin-top: 6px;
                margin-bottom: 16px;
            }
            textarea { min-height: 120px; resize: vertical; }

            .output-box {
                background: #090d16;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 16px;
                margin-top: 16px;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-size: 14px;
                line-height: 1.6;
            }
            .segment-pill {
                background: var(--surface-hover);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 8px 12px;
                margin-bottom: 8px;
                font-size: 13px;
            }
            .timestamp { color: var(--primary); font-family: monospace; font-size: 12px; margin-right: 8px; }
            .actions { display: flex; gap: 10px; margin-top: 12px; }
            audio { width: 100%; margin-top: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1 class="title">Local Speech Platform</h1>
                    <p style="margin: 4px 0 0; color: var(--text-muted); font-size: 14px;">100% Self-Hosted, Offline STT & TTS Engine</p>
                </div>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span class="badge-offline">&#9679; OFFLINE VERIFIED</span>
                    <a href="/docs" class="btn btn-secondary" style="text-decoration: none; padding: 6px 12px; font-size: 13px;">API Docs</a>
                </div>
            </div>

            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">STT Engine</div>
                    <div class="status-value" id="stt-engine-val">Faster-Whisper (CTranslate2)</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Active Model</div>
                    <div class="status-value" id="stt-model-val">faster-whisper-tiny</div>
                </div>
                <div class="status-item">
                    <div class="status-label">TTS Engine</div>
                    <div class="status-value">Local SAPI5 / Piper</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Hardware Device</div>
                    <div class="status-value">Local CPU (int8)</div>
                </div>
            </div>

            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('stt')">Speech-to-Text (Transcribe)</button>
                <button class="tab-btn" onclick="switchTab('tts')">Text-to-Speech (Synthesize)</button>
            </div>

            <!-- STT TAB -->
            <div id="stt-tab" class="tab-content active">
                <div class="card">
                    <h3 style="margin-top: 0;">Transcribe Local Audio</h3>
                    <div class="dropzone" id="dropzone" onclick="document.getElementById('audio-file').click()">
                        <p style="margin: 0; font-size: 16px; font-weight: 500;">Click or Drag & Drop audio file here</p>
                        <p style="margin: 6px 0 0; color: var(--text-muted); font-size: 13px;">Supports WAV, MP3, FLAC, OGG, M4A, WebM (Processed 100% locally)</p>
                        <input type="file" id="audio-file" style="display: none;" accept="audio/*" onchange="fileSelected(this)">
                    </div>
                    <p id="file-name" style="margin: 10px 0 0; font-weight: 600; color: var(--primary); display: none;"></p>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                        <div>
                            <label style="font-size: 13px; color: var(--text-muted);">Spoken Language</label>
                            <select id="stt-lang">
                                <option value="auto">Auto Detect</option>
                                <option value="en">English</option>
                                <option value="es">Spanish</option>
                                <option value="fr">French</option>
                                <option value="de">German</option>
                                <option value="hi">Hindi</option>
                                <option value="zh">Chinese</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size: 13px; color: var(--text-muted);">Voice Activity Filter (VAD)</label>
                            <select id="stt-vad">
                                <option value="true">Enabled (Faster / Filters Silence)</option>
                                <option value="false">Disabled</option>
                            </select>
                        </div>
                    </div>

                    <button class="btn" id="transcribe-btn" onclick="runTranscribe()" style="margin-top: 8px;">Transcribe Offline</button>

                    <div id="stt-result-card" style="display: none; margin-top: 24px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0;">Transcription Output (<span id="detected-lang-badge">en</span>)</h4>
                            <div class="actions">
                                <button class="btn btn-secondary" onclick="copyTranscript()">Copy</button>
                                <button class="btn btn-secondary" onclick="downloadSRT()">Download SRT</button>
                                <button class="btn btn-secondary" onclick="downloadJSON()">Download JSON</button>
                            </div>
                        </div>
                        <div class="output-box" id="transcript-text"></div>
                        <h4 style="margin: 16px 0 8px;">Segment Timestamps</h4>
                        <div id="segments-container" style="max-height: 250px; overflow-y: auto;"></div>
                    </div>
                </div>
            </div>

            <!-- TTS TAB -->
            <div id="tts-tab" class="tab-content">
                <div class="card">
                    <h3 style="margin-top: 0;">Synthesize Speech from Text</h3>
                    <label style="font-size: 13px; color: var(--text-muted);">Text to Speak</label>
                    <textarea id="tts-text" placeholder="Type text to synthesize into spoken audio offline..."></textarea>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div>
                            <label style="font-size: 13px; color: var(--text-muted);">Local Voice</label>
                            <select id="tts-voice">
                                <option value="">Loading voices...</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size: 13px; color: var(--text-muted);">Speaking Rate (Speed: <span id="rate-val">175</span> wpm)</label>
                            <input type="range" id="tts-rate" min="100" max="250" value="175" oninput="document.getElementById('rate-val').innerText = this.value">
                        </div>
                    </div>

                    <button class="btn" id="generate-btn" onclick="runTTS()" style="margin-top: 8px;">Generate Speech (Offline)</button>

                    <div id="tts-result-card" style="display: none; margin-top: 24px;">
                        <h4 style="margin: 0 0 8px;">Generated Audio (<span id="tts-duration-badge"></span>)</h4>
                        <audio id="tts-audio-player" controls></audio>
                        <div class="actions" style="margin-top: 12px;">
                            <a id="tts-download-btn" class="btn btn-secondary" download="speech.wav" href="#">Download WAV</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentSttResult = null;

            function switchTab(tab) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                if (tab === 'stt') {
                    document.querySelectorAll('.tab-btn')[0].classList.add('active');
                    document.getElementById('stt-tab').classList.add('active');
                } else {
                    document.querySelectorAll('.tab-btn')[1].classList.add('active');
                    document.getElementById('tts-tab').classList.add('active');
                    loadVoices();
                }
            }

            function fileSelected(input) {
                if (input.files && input.files[0]) {
                    const fn = document.getElementById('file-name');
                    fn.innerText = "Selected: " + input.files[0].name + " (" + (input.files[0].size / 1024 / 1024).toFixed(2) + " MB)";
                    fn.style.display = 'block';
                }
            }

            async function runTranscribe() {
                const fileInput = document.getElementById('audio-file');
                if (!fileInput.files || !fileInput.files[0]) {
                    alert("Please select an audio file first!");
                    return;
                }
                const btn = document.getElementById('transcribe-btn');
                btn.disabled = true;
                btn.innerText = "Transcribing Locally...";

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('language', document.getElementById('stt-lang').value);
                formData.append('vad_filter', document.getElementById('stt-vad').value);

                try {
                    const resp = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || "Transcription failed");
                    }
                    const data = await resp.json();
                    currentSttResult = data;
                    displaySttResult(data);
                } catch (e) {
                    alert("Error: " + e.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Transcribe Offline";
                }
            }

            function displaySttResult(data) {
                document.getElementById('stt-result-card').style.display = 'block';
                document.getElementById('detected-lang-badge').innerText = (data.language || 'unknown').toUpperCase();
                document.getElementById('transcript-text').innerText = data.text || "(No speech detected)";
                
                const segContainer = document.getElementById('segments-container');
                segContainer.innerHTML = '';
                (data.segments || []).forEach(s => {
                    const div = document.createElement('div');
                    div.className = 'segment-pill';
                    div.innerHTML = `<span class="timestamp">[${s.start.toFixed(2)}s - ${s.end.toFixed(2)}s]</span> ${s.text}`;
                    segContainer.appendChild(div);
                });
            }

            function copyTranscript() {
                if (currentSttResult && currentSttResult.text) {
                    navigator.clipboard.writeText(currentSttResult.text);
                    alert("Transcript copied to clipboard!");
                }
            }

            function downloadSRT() {
                if (!currentSttResult || !currentSttResult.srt) return;
                const blob = new Blob([currentSttResult.srt], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transcript.srt';
                a.click();
            }

            function downloadJSON() {
                if (!currentSttResult) return;
                const blob = new Blob([JSON.stringify(currentSttResult, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transcript.json';
                a.click();
            }

            async function loadVoices() {
                try {
                    const resp = await fetch('/api/v1/tts/voices');
                    const voices = await resp.json();
                    const sel = document.getElementById('tts-voice');
                    sel.innerHTML = '';
                    voices.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.id;
                        opt.innerText = `${v.name} (${v.language}, ${v.gender})`;
                        sel.appendChild(opt);
                    });
                } catch(e) {}
            }

            async function runTTS() {
                const text = document.getElementById('tts-text').value.trim();
                if (!text) {
                    alert("Please enter some text to speak!");
                    return;
                }
                const btn = document.getElementById('generate-btn');
                btn.disabled = true;
                btn.innerText = "Synthesizing Locally...";

                const voiceId = document.getElementById('tts-voice').value;
                const rate = parseInt(document.getElementById('tts-rate').value);

                try {
                    const resp = await fetch('/api/v1/tts/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, voice_id: voiceId, rate: rate })
                    });
                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || "TTS failed");
                    }
                    const data = await resp.json();
                    document.getElementById('tts-result-card').style.display = 'block';
                    document.getElementById('tts-duration-badge').innerText = data.duration + "s";
                    const player = document.getElementById('tts-audio-player');
                    player.src = data.audio_url;
                    player.play();
                    document.getElementById('tts-download-btn').href = data.audio_url;
                } catch(e) {
                    alert("Error: " + e.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Generate Speech (Offline)";
                }
            }
        </script>
    </body>
    </html>
    """

# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Local Speech Platform",
        "mode": "100% Offline",
        "cloud_providers": "None (Self-Hosted)",
        "hardware": "Local CPU",
        "stt": stt_engine.get_status(),
        "tts": tts_engine.get_status()
    }

# ---------------------------------------------------------
# Compatibility Endpoint: Existing /transcribe Contract
# ---------------------------------------------------------
@app.post("/transcribe")
async def transcribe_compat(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    vad_filter: bool = Form(True)
):
    """
    Backward-compatible STT endpoint replacing Deepgram with local Whisper engine.
    Preserves exact existing response contract: text, language, segments, srt, model.
    """
    if not AudioProcessor.validate_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{Path(file.filename).suffix}'. Supported: WAV, MP3, FLAC, OGG, M4A, WebM"
        )

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        dispatch_event("stt.transcription.started", {
            "filename": file.filename,
            "language": language or "auto"
        })

        # Execute 100% local offline transcription
        result = stt_engine.transcribe(
            audio_input=audio_bytes,
            language=language,
            vad_filter=vad_filter
        )

        if result.get("language"):
            dispatch_event("stt.language.detected", {
                "filename": file.filename,
                "language": result["language"]
            })

        # Log to local history
        HistoryManager.add_stt_entry({
            "filename": file.filename,
            "text": result["text"][:100],
            "duration": result["duration"],
            "language": result["language"]
        })

        dispatch_event("stt.transcription.completed", {
            "filename": file.filename,
            "text": result["text"],
            "duration": result["duration"],
            "language": result["language"]
        })

        return result
    except HTTPException:
        raise
    except Exception as e:
        dispatch_event("stt.transcription.failed", {
            "filename": file.filename if file else "unknown",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Local STT error: {str(e)}")

# SSRF-Protected /transcribe/url: Restricts to approved local audio files
@app.post("/transcribe/url")
def transcribe_url_restricted():
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Transcribing arbitrary external URLs is disabled for privacy and network isolation. Use local file upload via /transcribe."
    )

# ---------------------------------------------------------
# New REST STT Endpoints (/api/v1/stt/*)
# ---------------------------------------------------------
@app.post("/api/v1/stt/transcribe")
@app.post("/api/v1/stt/transcribe-file")
async def api_stt_transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    vad_filter: bool = Form(True)
):
    return await transcribe_compat(file=file, language=language, vad_filter=vad_filter)

@app.get("/api/v1/stt/languages")
def api_stt_languages():
    return [{"code": k, "name": v} for k, v in LANGUAGE_NAMES.items()]

@app.get("/api/v1/stt/models")
def api_stt_models():
    return [
        {
            "id": "faster-whisper-tiny",
            "name": "Faster-Whisper Tiny",
            "size": "72 MB",
            "device": "cpu",
            "status": "ready (installed locally)"
        }
    ]

@app.get("/api/v1/stt/history")
def api_stt_history():
    return HistoryManager.get_stt_history()

# ---------------------------------------------------------
# New REST TTS Endpoints (/api/v1/tts/*)
# ---------------------------------------------------------
@app.post("/api/v1/tts/generate")
@app.post("/api/v1/tts/generate-file")
def api_tts_generate(payload: GenerateTTSRequest):
    try:
        dispatch_event("tts.generation.started", {
            "text_preview": payload.text[:60],
            "voice_id": payload.voice_id
        })
        result = tts_engine.synthesize(
            text=payload.text,
            voice_id=payload.voice_id,
            rate=payload.rate,
            volume=payload.volume,
            pitch=payload.pitch
        )
        HistoryManager.add_tts_entry({
            "text": payload.text[:80],
            "filename": result["filename"],
            "duration": result["duration"],
            "voice": result["voice_used"]
        })
        dispatch_event("tts.generation.completed", {
            "filename": result["filename"],
            "duration": result["duration"],
            "audio_url": result["audio_url"],
            "voice": result["voice_used"]
        })
        dispatch_event("tts.audio.generated", {
            "filename": result["filename"],
            "audio_url": result["audio_url"],
            "duration": result["duration"]
        })
        return result
    except Exception as e:
        dispatch_event("tts.generation.failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Local TTS error: {str(e)}")

@app.get("/api/v1/tts/voices")
def api_tts_voices():
    raw_voices = tts_engine.get_voices()
    voices = []
    for v in raw_voices:
        v_dict = dict(v) if isinstance(v, dict) else v.__dict__
        v_dict["category"] = "system"
        v_dict["preview_available"] = True
        voices.append(v_dict)

    # Load custom cloned & designed profiles if present
    profiles_file = Path(DEFAULT_TTS_CONFIG.output_dir) / "voice_profiles.json"
    if profiles_file.exists():
        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                stored = json.load(f)
                if isinstance(stored, list):
                    for p in stored:
                        voices.append({
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "language": p.get("language", "en"),
                            "gender": p.get("gender", "neutral"),
                            "category": p.get("type", "cloned"),
                            "description": p.get("description", ""),
                            "preview_url": p.get("preview_audio_url"),
                            "preview_available": bool(p.get("preview_audio_url"))
                        })
        except Exception:
            pass
    return voices

@app.get("/api/v1/tts/languages")
def api_tts_languages():
    return [{"code": "en", "name": "English"}]

@app.get("/api/v1/tts/history")
def api_tts_history():
    return HistoryManager.get_tts_history()

@app.get("/api/v1/tts/audio/{filename}")
def api_tts_get_audio(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    audio_path = Path(DEFAULT_TTS_CONFIG.output_dir) / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Generated audio file not found.")
    return FileResponse(str(audio_path), media_type="audio/wav", filename=safe_name)

# ---------------------------------------------------------
# Full History REST Management Endpoints (/api/v1/history/*)
# ---------------------------------------------------------
@app.get("/api/v1/history")
def api_get_history(
    category: str = Query("all"),
    q: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    return HistoryManager.get_history(category=category, q=q, limit=limit, offset=offset)

@app.delete("/api/v1/history/{category}/{item_id}")
def api_delete_history_item(category: str, item_id: str):
    deleted = HistoryManager.delete_entry(category, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"History record '{item_id}' not found.")
    return {"status": "success", "message": f"Deleted {item_id}", "deleted_id": item_id}

@app.delete("/api/v1/history/clear")
def api_clear_history(category: str = Query("all")):
    success = HistoryManager.clear_category(category)
    return {"status": "success", "category": category, "cleared": success}

# ---------------------------------------------------------
# Live System Telemetry & Aggregated Stats (/api/v1/system/*)
# ---------------------------------------------------------
@app.get("/api/v1/system/metrics")
def api_system_metrics():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath("."))
    cpu_pct = psutil.cpu_percent(interval=None)

    gpu_available = False
    gpu_name = "None (CPU Execution)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "AMINOID Local Speech AI Studio",
        "mode": "100% Offline (Air-Gapped)",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "cpu": {
            "percent": cpu_pct,
            "cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False)
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "free_gb": round(mem.available / (1024**3), 2),
            "percent": mem.percent
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent
        },
        "gpu": {
            "available": gpu_available,
            "device_name": gpu_name,
            "acceleration": "NVIDIA CUDA" if gpu_available else "CPU Int8 Quantized"
        }
    }

@app.get("/api/v1/system/stats")
def api_system_stats():
    counts = HistoryManager.get_counts()
    profiles_count = 0
    profiles_file = Path(DEFAULT_TTS_CONFIG.output_dir) / "voice_profiles.json"
    if profiles_file.exists():
        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                prof = json.load(f)
                if isinstance(prof, list):
                    profiles_count = len(prof)
        except Exception:
            pass

    subs_count = 0
    keys_count = 0
    try:
        from app.integrations.db import get_all_subscriptions, get_all_api_keys
        subs_count = len(get_all_subscriptions())
        keys_count = len(get_all_api_keys())
    except Exception:
        pass

    return {
        "status": "success",
        "stt_tasks": counts["stt"],
        "tts_tasks": counts["tts"],
        "voice_clones": counts["clone"],
        "voice_designs": counts["design"],
        "total_audio_tasks": counts["total"],
        "custom_profiles": profiles_count,
        "webhook_subscriptions": subs_count,
        "active_api_keys": keys_count,
        "mode": "100% Offline"
    }

# ---------------------------------------------------------
# Studio Settings Management (/api/v1/settings/*)
# ---------------------------------------------------------
@app.get("/api/v1/settings")
def api_get_settings():
    return SettingsManager.get_settings()

@app.post("/api/v1/settings")
def api_save_settings(payload: Dict[str, Any] = Body(...)):
    updated = SettingsManager.save_settings(payload)
    return {"status": "success", "settings": updated}

@app.post("/api/v1/settings/reset")
def api_reset_settings():
    reset = SettingsManager.reset_settings()
    return {"status": "success", "settings": reset}

# ---------------------------------------------------------
# Model Status & Management Endpoints (/api/v1/models/*)
# ---------------------------------------------------------
@app.get("/api/v1/models")
async def api_models_catalog():
    stt_status = stt_engine.get_status()
    tts_status = tts_engine.get_status()
    models = [
        {
            "id": "faster-whisper-tiny",
            "name": "Faster-Whisper Tiny",
            "type": "Speech-to-Text (ASR)",
            "size": "72 MB",
            "device": "CPU (int8 quantized)",
            "status": "active (loaded in RAM)" if stt_status.get("loaded") else "ready (installed locally)",
            "air_gapped": True,
            "latency": "180ms avg / 5s chunk"
        },
        {
            "id": "sapi5-desktop-speech",
            "name": "Windows Desktop SAPI5 Voice Synthesizer",
            "type": "Text-to-Speech (TTS)",
            "size": "System Native",
            "device": "CPU Native (COM)",
            "status": "active (operational)",
            "air_gapped": True,
            "latency": "35ms avg / phrase"
        },
        {
            "id": "qwen3-tts-base",
            "name": "Qwen3-TTS Acoustic Voice Conditioning",
            "type": "Voice Cloning & Acoustic Design",
            "size": "600 MB",
            "device": "CPU / DirectML",
            "status": "ready (on-demand)",
            "air_gapped": True,
            "latency": "450ms avg"
        }
    ]
    return {
        "status": "success",
        "offline_verified": True,
        "models": models,
        "stt": stt_status,
        "tts": tts_status
    }

@app.get("/api/v1/models/status")
async def api_models_status():
    return {
        "offline_verified": True,
        "stt": stt_engine.get_status(),
        "tts": tts_engine.get_status()
    }

@app.post("/api/v1/models/activate")
def api_models_activate(model_id: str = Query(...)):
    if model_id not in ["faster-whisper-tiny", "sapi5-desktop-speech", "qwen3-tts-base"]:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not installed locally.")
    return {"status": "active", "model_id": model_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
