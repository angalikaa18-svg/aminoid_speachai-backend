"""
FastAPI Router for AMINOID Speech Integrations, External APIs, and Webhooks.
Mounted at /api/v1/integrations.
"""

import os
import time
import uuid
import socket
import secrets
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    HTTPException,
    Query,
    Request,
    Depends,
    status
)
from fastapi.responses import JSONResponse, FileResponse
import httpx

from .schemas import (
    PlatformCreateRequest,
    PlatformUpdateRequest,
    ApiKeyCreateRequest,
    WebhookSubscriptionCreateRequest,
    WebhookSubscriptionUpdateRequest,
    TestWebhookRequest,
    TestConnectionRequest,
    ExternalTTSRequest
)
from .db import (
    create_integration,
    get_integration,
    list_integrations,
    update_integration,
    delete_integration,
    list_api_keys,
    toggle_api_key,
    revoke_api_key,
    create_webhook_subscription,
    get_webhook_subscription,
    list_webhook_subscriptions,
    update_webhook_subscription,
    delete_webhook_subscription,
    list_webhook_deliveries,
    get_webhook_delivery,
    get_integration_stats,
    log_api_request,
    log_webhook_delivery
)
from .auth import (
    generate_api_key,
    RequireApiKey,
    OptionalApiKey,
    SUPPORTED_PERMISSIONS
)
from .dispatcher import (
    dispatch_event,
    send_test_webhook,
    _deliver_to_subscriber
)

from app.stt.engine import LocalSTTEngine
from app.stt.config import DEFAULT_STT_CONFIG
from app.tts.engine import LocalTTSEngine
from app.tts.config import DEFAULT_TTS_CONFIG
from app.history import HistoryManager

router = APIRouter(prefix="/api/v1/integrations", tags=["External Platform Integrations"])

# Local speech engine singletons
_stt_engine = LocalSTTEngine()
_tts_engine = LocalTTSEngine()

def get_detected_local_ip() -> str:
    """Detects local LAN IPv4 address without external internet dependency."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

SUPPORTED_WEBHOOK_EVENTS = [
    {"id": "stt.completed", "label": "STT Transcription Completed", "category": "Speech", "description": "Triggered when audio is transcribed into text by local STT engine."},
    {"id": "stt.failed", "label": "STT Transcription Failed", "category": "Speech", "description": "Triggered when speech-to-text processing encounters an error or empty audio."},
    {"id": "tts.started", "label": "TTS Synthesis Started", "category": "Speech", "description": "Triggered when text synthesis begins."},
    {"id": "tts.completed", "label": "TTS Synthesis Completed", "category": "Speech", "description": "Triggered when audio is synthesized and written to disk."},
    {"id": "tts.failed", "label": "TTS Synthesis Failed", "category": "Speech", "description": "Triggered when voice synthesis fails."},
    {"id": "audio.generated", "label": "Audio File Ready", "category": "Speech", "description": "Triggered whenever new WAV audio is available for download or streaming."},
    {"id": "model.loaded", "label": "Model Loaded", "category": "Model", "description": "Triggered when a speech model is loaded into memory."},
    {"id": "model.status_changed", "label": "Model Status Changed", "category": "Model", "description": "Triggered when STT or TTS engine status changes."},
    {"id": "integration.connected", "label": "Integration Connected", "category": "Integration", "description": "Triggered when an external platform connection is verified."},
    {"id": "integration.failed", "label": "Integration Connection Failed", "category": "Integration", "description": "Triggered when health ping to external platform fails."},
    {"id": "api.request.completed", "label": "External API Request Success", "category": "API", "description": "Triggered upon successful external API invocation."},
    {"id": "api.request.failed", "label": "External API Request Failed", "category": "API", "description": "Triggered upon failed external API call or auth error."},
    {"id": "callzenza.session.started", "label": "CallZenza Session Started", "category": "CallZenza", "description": "Triggered when CallZenza initiates speech session with AMINOID."},
    {"id": "callzenza.audio.processed", "label": "CallZenza Audio Processed", "category": "CallZenza", "description": "Triggered when CallZenza audio packet finishes processing."}
]

# -----------------------------------------------------------------------------
# 1. Overview & Statistics
# -----------------------------------------------------------------------------
@router.get("/overview")
def get_overview():
    """Returns live backend system status, engine statistics, and network topology."""
    stats = get_integration_stats()
    local_ip = get_detected_local_ip()
    port = 8080

    voices = _tts_engine.get_voices()
    
    return {
        "success": True,
        "backend_connected": True,
        "speech_engine_status": "Ready",
        "stt_model_status": {
            "model_name": DEFAULT_STT_CONFIG.model_name,
            "device": DEFAULT_STT_CONFIG.device,
            "status": "ready"
        },
        "tts_model_status": {
            "engine_name": DEFAULT_TTS_CONFIG.engine_name,
            "status": "ready",
            "voices_available": len(voices)
        },
        "voices_count": len(voices),
        "statistics": stats,
        "network": {
            "local_ip": local_ip,
            "port": port,
            "localhost_url": f"http://localhost:{port}",
            "lan_url": f"http://{local_ip}:{port}",
            "is_lan": local_ip != "127.0.0.1",
            "warning": "When integrating with cloud-hosted platforms or external webhooks, ensure either your local network IP is routable or a public domain/tunnel is configured."
        },
        "supported_events": SUPPORTED_WEBHOOK_EVENTS
    }

# -----------------------------------------------------------------------------
# 2. External Platforms CRUD
# -----------------------------------------------------------------------------
@router.get("/platforms")
def list_connected_platforms():
    """Lists all configured external platform integrations."""
    return {"success": True, "platforms": list_integrations()}

@router.post("/platforms")
def register_platform(payload: PlatformCreateRequest):
    """Registers a new external platform integration."""
    created = create_integration(
        name=payload.name,
        platform_type=payload.platform_type,
        description=payload.description,
        base_url=payload.base_url,
        webhook_url=payload.webhook_url,
        auth_method=payload.auth_method or "none",
        api_key_encrypted=payload.api_key,
        secret_key_encrypted=payload.secret_key,
        custom_headers=payload.custom_headers,
        timeout_sec=payload.timeout_sec or 30
    )
    dispatch_event("integration.connected", {
        "integration_id": created["id"],
        "name": created["name"],
        "platform_type": created["platform_type"]
    })
    return {"success": True, "platform": created}

@router.get("/platforms/{platform_id}")
def get_platform_details(platform_id: str):
    platform = get_integration(platform_id)
    if not platform:
        raise HTTPException(status_code=404, detail="Integration not found")
    return {"success": True, "platform": platform}

@router.put("/platforms/{platform_id}")
def update_platform_config(platform_id: str, payload: PlatformUpdateRequest):
    existing = get_integration(platform_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    update_data = payload.model_dump(exclude_unset=True)
    if "api_key" in update_data:
        update_data["api_key_encrypted"] = update_data.pop("api_key")
    if "secret_key" in update_data:
        update_data["secret_key_encrypted"] = update_data.pop("secret_key")

    updated = update_integration(platform_id, update_data)
    return {"success": True, "platform": updated}

@router.delete("/platforms/{platform_id}")
def remove_platform(platform_id: str):
    existing = get_integration(platform_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Integration not found")
    delete_integration(platform_id)
    dispatch_event("integration.disconnected", {"integration_id": platform_id, "name": existing["name"]})
    return {"success": True, "message": "Integration removed successfully"}

@router.post("/platforms/test-url")
async def test_arbitrary_url_connection(payload: TestConnectionRequest):
    """Performs real backend ping to any target URL to verify connectivity."""
    return await _perform_connection_ping(
        url=payload.base_url,
        auth_method=payload.auth_method or "none",
        api_key=payload.api_key,
        headers=payload.custom_headers or {},
        timeout=10
    )

@router.post("/platforms/{platform_id}/test")
async def test_platform_connection(platform_id: str):
    """Tests real HTTP connectivity to an existing saved platform base_url."""
    platform = get_integration(platform_id)
    if not platform:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    target_url = platform.get("base_url")
    if not target_url:
        raise HTTPException(status_code=400, detail="Platform does not have a base_url configured.")

    return await _perform_connection_ping(
        url=target_url,
        auth_method=platform.get("auth_method", "none"),
        headers=platform.get("custom_headers", {}),
        timeout=platform.get("timeout_sec", 10)
    )

async def _perform_connection_ping(
    url: str,
    auth_method: str = "none",
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    req_headers = {"User-Agent": "AMINOID-Speech-Integration-Ping/1.0"}
    if headers and isinstance(headers, dict):
        req_headers.update(headers)
    if auth_method == "bearer_token" and api_key:
        req_headers["Authorization"] = f"Bearer {api_key}"
    elif auth_method == "api_key" and api_key:
        req_headers["X-API-Key"] = api_key

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=float(timeout), verify=False) as client:
            res = await client.get(url, headers=req_headers)
            duration_ms = (time.time() - t0) * 1000.0
            return {
                "success": res.status_code < 400,
                "http_status": res.status_code,
                "duration_ms": round(duration_ms, 2),
                "response_preview": res.text[:500],
                "target_url": url,
                "message": f"Successfully reached platform endpoint ({res.status_code}) in {duration_ms:.1f}ms"
            }
    except Exception as e:
        duration_ms = (time.time() - t0) * 1000.0
        return {
            "success": False,
            "http_status": None,
            "duration_ms": round(duration_ms, 2),
            "error": str(e),
            "target_url": url,
            "message": f"Connection check failed: {str(e)}"
        }

# -----------------------------------------------------------------------------
# 3. AMINOID API Keys
# -----------------------------------------------------------------------------
@router.get("/api-keys")
def list_developer_keys():
    """Lists developer API keys (with key prefix only, hashed secret is secured)."""
    return {"success": True, "keys": list_api_keys(), "supported_permissions": SUPPORTED_PERMISSIONS}

@router.post("/api-keys")
def create_developer_key(payload: ApiKeyCreateRequest):
    """Generates a new cryptographically secure API key."""
    raw_key, record = generate_api_key(
        name=payload.name,
        permissions=payload.permissions,
        expires_in_days=payload.expires_in_days
    )
    return {
        "success": True,
        "api_key": raw_key, # Returned ONLY ONCE
        "key_record": record,
        "message": "API key generated. Store this key securely; it will not be displayed again."
    }

@router.patch("/api-keys/{key_id}/toggle")
def toggle_developer_key(key_id: str, is_active: bool = Query(...)):
    ok = toggle_api_key(key_id, is_active)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": f"API key {'activated' if is_active else 'deactivated'} successfully"}

@router.delete("/api-keys/{key_id}")
def revoke_developer_key(key_id: str):
    ok = revoke_api_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": "API key revoked permanently"}

# -----------------------------------------------------------------------------
# 4. Webhook Subscriptions & Deliveries
# -----------------------------------------------------------------------------
@router.get("/webhooks")
def list_subscriptions():
    """Returns all configured webhook subscriptions."""
    return {"success": True, "subscriptions": list_webhook_subscriptions(), "supported_events": SUPPORTED_WEBHOOK_EVENTS}

@router.post("/webhooks")
def create_subscription(payload: WebhookSubscriptionCreateRequest):
    """Creates a new webhook event subscription with HMAC-SHA256 secret."""
    secret = payload.secret_key or f"whsec_{secrets.token_hex(16)}"
    created = create_webhook_subscription(
        name=payload.name,
        target_url=payload.target_url,
        events=payload.events,
        secret_key=secret,
        integration_id=payload.integration_id,
        custom_headers=payload.custom_headers,
        retry_count=payload.retry_count or 3,
        retry_interval_sec=payload.retry_interval_sec or 5,
        timeout_sec=payload.timeout_sec or 10
    )
    return {"success": True, "subscription": created}

@router.post("/webhooks/test")
async def test_webhook_dispatch(payload: TestWebhookRequest):
    """Dispatches a real HTTP test webhook event with authentic HMAC signature."""
    res = await send_test_webhook(
        target_url=payload.target_url,
        secret_key=payload.secret_key or "whsec_test_demo",
        event_name=payload.event_name or "test.ping",
        sample_payload=payload.sample_payload,
        custom_headers=payload.custom_headers
    )
    return res

@router.get("/webhooks/deliveries")
def list_deliveries(
    limit: int = Query(50, ge=1, le=200),
    event_name: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Returns recent webhook delivery logs with filter options."""
    clean_event = event_name.strip() if event_name and event_name.strip() else None
    clean_status = status.strip() if status and status.strip() else None
    deliveries = list_webhook_deliveries(limit=limit, event_name=clean_event, status=clean_status)
    return {"success": True, "deliveries": deliveries}

@router.get("/webhooks/deliveries/{delivery_id}")
def get_delivery_details(delivery_id: str):
    deliv = get_webhook_delivery(delivery_id)
    if not deliv:
        raise HTTPException(status_code=404, detail="Delivery log not found")
    return {"success": True, "delivery": deliv}

@router.post("/webhooks/deliveries/{delivery_id}/retry")
async def retry_webhook_delivery(delivery_id: str):
    """Retries a previously logged webhook delivery."""
    deliv = get_webhook_delivery(delivery_id)
    if not deliv:
        raise HTTPException(status_code=404, detail="Delivery log not found")
    
    sub = get_webhook_subscription(deliv["subscription_id"])
    if not sub:
        # Re-deliver directly using delivery target_url
        sub = {
            "id": deliv["subscription_id"],
            "target_url": deliv["target_url"],
            "secret_key": "retry_secret",
            "retry_count": 1,
            "retry_interval_sec": 1,
            "timeout_sec": 10,
            "custom_headers": {}
        }

    res = await _deliver_to_subscriber(sub, deliv["event_name"], deliv["payload"])
    return {"success": res["status"] == "success", "result": res}

@router.get("/webhooks/{subscription_id}")
def get_subscription(subscription_id: str):
    sub = get_webhook_subscription(subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Webhook subscription not found")
    return {"success": True, "subscription": sub}

@router.put("/webhooks/{subscription_id}")
def update_subscription(subscription_id: str, payload: WebhookSubscriptionUpdateRequest):
    existing = get_webhook_subscription(subscription_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Webhook subscription not found")
    
    update_data = payload.model_dump(exclude_unset=True)
    updated = update_webhook_subscription(subscription_id, update_data)
    return {"success": True, "subscription": updated}

@router.delete("/webhooks/{subscription_id}")
def delete_subscription(subscription_id: str):
    existing = get_webhook_subscription(subscription_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Webhook subscription not found")
    delete_webhook_subscription(subscription_id)
    return {"success": True, "message": "Webhook subscription deleted successfully"}

# -----------------------------------------------------------------------------
# 5. External Speech APIs (Real STT, TTS & Voices)
# -----------------------------------------------------------------------------
@router.get("/voices")
def get_external_voices(auth: Optional[Dict[str, Any]] = Depends(OptionalApiKey())):
    """Exposes real voices available in the AMINOID Speech Engine."""
    voices = _tts_engine.get_voices()
    formatted = [
        {
            "voice_id": v.id,
            "name": v.name,
            "language": v.language,
            "gender": v.gender,
            "available": True
        }
        for v in voices
    ]
    return {
        "success": True,
        "count": len(formatted),
        "voices": formatted
    }

@router.post("/stt/transcribe")
async def external_stt_transcribe(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
    vad_filter: bool = Form(True),
    word_timestamps: bool = Form(False),
    auth: Optional[Dict[str, Any]] = Depends(OptionalApiKey())
):
    """
    Exposes real local Whisper STT engine to external platforms.
    Accepts multipart upload under 'audio' or 'file'.
    Supports WAV, MP3, M4A, FLAC, WebM, OGG.
    """
    upload = audio or file
    if not upload:
        raise HTTPException(status_code=400, detail="Missing audio file. Provide via 'audio' or 'file' form field.")

    t0 = time.time()
    req_id = f"stt_{uuid.uuid4().hex[:10]}"
    client_ip = request.client.host if request.client else "127.0.0.1"

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        # Execute local Whisper inference directly on in-memory audio bytes
        result = _stt_engine.transcribe(
            audio_input=content,
            language=language if language and language.lower() != "auto" else None,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
        proc_time = round(time.time() - t0, 2)

        # Log to History & API Request Logs
        HistoryManager.add_stt_entry({
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
            "model": result.get("model", DEFAULT_STT_CONFIG.model_name)
        })
        log_api_request(
            endpoint="/api/v1/integrations/stt/transcribe",
            method="POST",
            status_code=200,
            processing_time_ms=proc_time * 1000.0,
            api_key_id=getattr(request.state, "api_key_id", None),
            client_ip=client_ip
        )

        # Dispatch background webhook event
        dispatch_event("stt.completed", {
            "request_id": req_id,
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
            "processing_time": proc_time,
            "model": result.get("model", DEFAULT_STT_CONFIG.model_name)
        })

        return {
            "success": True,
            "request_id": req_id,
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
            "processing_time": proc_time,
            "segments": result.get("segments", []),
            "model": result.get("model", DEFAULT_STT_CONFIG.model_name)
        }

    except HTTPException:
        raise
    except Exception as e:
        proc_time = round(time.time() - t0, 2)
        log_api_request(
            endpoint="/api/v1/integrations/stt/transcribe",
            method="POST",
            status_code=500,
            processing_time_ms=proc_time * 1000.0,
            api_key_id=getattr(request.state, "api_key_id", None),
            client_ip=client_ip
        )
        dispatch_event("stt.failed", {"request_id": req_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Speech-to-Text inference failed: {str(e)}")

@router.post("/tts/generate")
def external_tts_generate(
    request: Request,
    payload: ExternalTTSRequest,
    auth: Optional[Dict[str, Any]] = Depends(OptionalApiKey())
):
    """
    Exposes real local TTS engine to external platforms.
    Synthesizes speech into WAV using selected voice_id.
    Returns JSON with audio_url or raw audio/wav stream.
    """
    t0 = time.time()
    req_id = f"tts_{uuid.uuid4().hex[:10]}"
    client_ip = request.client.host if request.client else "127.0.0.1"

    try:
        dispatch_event("tts.started", {"request_id": req_id, "text_preview": payload.text[:60]})
        
        result = _tts_engine.synthesize(
            text=payload.text,
            voice_id=payload.voice_id,
            rate=payload.rate,
            volume=payload.volume,
            pitch=payload.pitch
        )
        proc_time = round(time.time() - t0, 2)

        # Log to History & API Request Logs
        HistoryManager.add_tts_entry({
            "text": payload.text[:80],
            "filename": result["filename"],
            "duration": result["duration"],
            "voice": result.get("voice_used", "Default")
        })
        log_api_request(
            endpoint="/api/v1/integrations/tts/generate",
            method="POST",
            status_code=200,
            processing_time_ms=proc_time * 1000.0,
            api_key_id=getattr(request.state, "api_key_id", None),
            client_ip=client_ip
        )

        dispatch_event("tts.completed", {
            "request_id": req_id,
            "filename": result["filename"],
            "duration": result["duration"],
            "audio_url": result["audio_url"],
            "voice": result.get("voice_used", "Default")
        })
        dispatch_event("audio.generated", {
            "filename": result["filename"],
            "audio_url": result["audio_url"],
            "duration": result["duration"]
        })

        # Return raw audio stream if client requested response_mode='audio'
        if payload.response_mode and payload.response_mode.lower() == "audio":
            file_path = Path(DEFAULT_TTS_CONFIG.output_dir) / result["filename"]
            if file_path.exists():
                return FileResponse(
                    path=str(file_path),
                    media_type="audio/wav",
                    filename=result["filename"]
                )

        return {
            "success": True,
            "request_id": req_id,
            "audio_url": result["audio_url"],
            "full_audio_url": f"http://127.0.0.1:8080{result['audio_url']}",
            "filename": result["filename"],
            "duration": result["duration"],
            "voice_id": payload.voice_id,
            "voice_used": result.get("voice_used"),
            "format": "wav",
            "processing_time": proc_time
        }

    except Exception as e:
        proc_time = round(time.time() - t0, 2)
        log_api_request(
            endpoint="/api/v1/integrations/tts/generate",
            method="POST",
            status_code=500,
            processing_time_ms=proc_time * 1000.0,
            api_key_id=getattr(request.state, "api_key_id", None),
            client_ip=client_ip
        )
        dispatch_event("tts.failed", {"request_id": req_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Text-to-Speech generation failed: {str(e)}")
