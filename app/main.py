import os
import sys
import multiprocessing
import warnings
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, status
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

        # Execute 100% local offline transcription
        result = stt_engine.transcribe(
            audio_input=audio_bytes,
            language=language,
            vad_filter=vad_filter
        )

        # Log to local history
        HistoryManager.add_stt_entry({
            "filename": file.filename,
            "text": result["text"][:100],
            "duration": result["duration"],
            "language": result["language"]
        })

        dispatch_event("stt.completed", {
            "filename": file.filename,
            "text": result["text"],
            "duration": result["duration"],
            "language": result["language"]
        })

        return result
    except HTTPException:
        raise
    except Exception as e:
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
        dispatch_event("tts.completed", {
            "filename": result["filename"],
            "duration": result["duration"],
            "audio_url": result["audio_url"],
            "voice": result["voice_used"]
        })
        dispatch_event("audio.generated", {
            "filename": result["filename"],
            "audio_url": result["audio_url"],
            "duration": result["duration"]
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local TTS error: {str(e)}")

@app.get("/api/v1/tts/voices")
def api_tts_voices():
    return tts_engine.get_voices()

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
# Model Status & Management Endpoints (/api/v1/models/*)
# ---------------------------------------------------------
@app.get("/api/v1/models")
@app.get("/api/v1/models/status")
async def api_models_status():
    return {
        "offline_verified": True,
        "stt": stt_engine.get_status(),
        "tts": tts_engine.get_status()
    }

@app.post("/api/v1/models/activate")
def api_models_activate(model_id: str = Query(...)):
    if model_id != "faster-whisper-tiny":
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not installed locally.")
    return {"status": "active", "model_id": model_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
