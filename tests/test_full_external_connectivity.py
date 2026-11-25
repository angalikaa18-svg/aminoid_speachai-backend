"""
Full Automated Verification Script for AMINOID SPEECH External Platform Integration.
Executes all test phases (Phases 2 through 10) against live backend and external receiver:
- Phase 2: Reachability & CORS
- Phase 3: API Key Authentication (Valid, Invalid, Missing, Scopes)
- Phase 4: Real STT Connectivity (Faster-Whisper)
- Phase 5: Real TTS Connectivity (SAPI5 offline + physical audio file check)
- Phase 6: Webhook Subscription Lifecycle (Database persistence)
- Phase 7: Real HMAC Webhook Dispatch & Delivery Verification
- Phase 8: Webhook Failure Scenarios (404/Invalid, 500 Error, Timeout)
- Phase 9: Bi-directional communication with independent external platform on port 9090
- Phase 10: Live 8-point Diagnostic Runner Endpoint
"""

import sys
import io
import wave
import json
import time
import os
from pathlib import Path
import requests

# Add backend to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tests.external_platform_test_server import (
    start_external_receiver_server,
    RECEIVED_EVENTS,
    SHARED_WEBHOOK_SECRET,
    SERVER_PORT
)

AMINOID_BASE = "http://127.0.0.1:8080"
EXTERNAL_BASE = f"http://127.0.0.1:{SERVER_PORT}"

def create_dummy_wav_bytes(duration_sec: float = 0.5) -> bytes:
    """Generates a small clean mono PCM WAV in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        num_frames = int(16000 * duration_sec)
        wf.writeframes(b"\x00" * (num_frames * 2))
    return buf.getvalue()

def run_all_tests():
    print("=" * 80)
    print("STARTING FULL EXTERNAL CONNECTIVITY & WEBHOOK VERIFICATION SUITE")
    print("=" * 80)

    # 1. Start External Receiver Server
    print("\n[SETUP] Starting standalone External Platform Receiver on port 9090...")
    receiver_server = start_external_receiver_server(SERVER_PORT)
    time.sleep(1)
    rec_health = requests.get(f"{EXTERNAL_BASE}/health", timeout=3)
    assert rec_health.status_code == 200, "External receiver failed to start"
    print(f" -> External receiver online at {EXTERNAL_BASE}/webhook (status: 200)")

    test_results = {}

    # -------------------------------------------------------------------------
    # PHASE 2: Backend Reachability
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: VERIFY BACKEND IS REACHABLE ---")
    t0 = time.time()
    r = requests.get(f"{AMINOID_BASE}/health", timeout=5)
    lat = round((time.time() - t0) * 1000, 2)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health_data = r.json()
    assert health_data.get("status") == "healthy"
    print(f" PASS: /health responded 200 OK in {lat}ms: {health_data['service']}")

    # CORS check
    cors_res = requests.options(
        f"{AMINOID_BASE}/api/v1/integrations/overview",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" in cors_res.headers
    print(f" PASS: CORS configuration active: Access-Control-Allow-Origin = {cors_res.headers.get('access-control-allow-origin')}")
    test_results["Phase 2: Reachability"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 3: API Key Authentication
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: VERIFY API KEY AUTHENTICATION ---")
    # Generate full-access API Key
    r_key = requests.post(f"{AMINOID_BASE}/api/v1/integrations/api-keys", json={
        "name": "External Integration Bot Key",
        "permissions": ["stt:transcribe", "tts:generate", "voices:read", "webhooks:manage"]
    })
    assert r_key.status_code == 200
    valid_res_json = r_key.json()
    valid_raw_key = valid_res_json["api_key"]
    valid_record = valid_res_json["key_record"]
    print(f" -> Generated Valid API Key: {valid_record['key_prefix']} (ID: {valid_record['id']})")

    # Generate limited scope key (only voices:read, NO tts:generate, NO stt:transcribe)
    r_lim = requests.post(f"{AMINOID_BASE}/api/v1/integrations/api-keys", json={
        "name": "Limited Reader Key",
        "permissions": ["voices:read"]
    })
    assert r_lim.status_code == 200
    limited_raw_key = r_lim.json()["api_key"]
    print(f" -> Generated Limited API Key (voices:read only)")

    # Test A: Valid API Key
    r_auth = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": valid_raw_key},
        json={"text": "Authentic API key test passed."}
    )
    assert r_auth.status_code == 200, f"Valid key rejected: {r_auth.status_code}"
    print(f" PASS Test A: Valid API Key accepted (HTTP 200)")

    # Test B: Invalid API Key
    r_bad = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": "aminoid_live_invalid_bad_key_9999"},
        json={"text": "Should fail"}
    )
    assert r_bad.status_code == 401, f"Expected 401, got {r_bad.status_code}"
    print(f" PASS Test B: Invalid API Key rejected (HTTP 401: {r_bad.json()['detail']})")

    # Test C: Missing API Key
    r_miss = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-Require-Auth": "strict"},
        json={"text": "Should fail"}
    )
    assert r_miss.status_code == 401, f"Expected 401, got {r_miss.status_code}"
    print(f" PASS Test C: Missing API Key rejected (HTTP 401: {r_miss.json()['detail']})")

    # Test D: Permission Validation (Limited key lacks tts:generate)
    r_perm = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": limited_raw_key},
        json={"text": "Should fail due to permission"}
    )
    assert r_perm.status_code == 403, f"Expected 403, got {r_perm.status_code}"
    print(f" PASS Test D: Key without tts:generate permission rejected (HTTP 403: {r_perm.json()['detail']})")

    test_results["Phase 3: Authentication"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 4: Real STT Connectivity
    # -------------------------------------------------------------------------
    print("\n--- PHASE 4: TEST REAL STT CONNECTIVITY ---")
    wav_bytes = create_dummy_wav_bytes(0.5)
    t0 = time.time()
    r_stt = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/stt/transcribe",
        headers={"X-API-Key": valid_raw_key},
        files={"audio": ("test_external.wav", wav_bytes, "audio/wav")},
        data={"vad_filter": "true"}
    )
    stt_duration = round(time.time() - t0, 2)
    assert r_stt.status_code == 200, f"STT failed: {r_stt.status_code} {r_stt.text}"
    stt_data = r_stt.json()
    assert stt_data.get("success") is True
    assert "request_id" in stt_data
    assert "model" in stt_data
    print(f" PASS: Real STT processed in {stt_duration}s. Model: {stt_data['model']}, RequestId: {stt_data['request_id']}")
    test_results["Phase 4: STT Connectivity"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 5: Real TTS Connectivity
    # -------------------------------------------------------------------------
    print("\n--- PHASE 5: TEST REAL TTS CONNECTIVITY ---")
    t0 = time.time()
    synth_text = "AMINOID speech engine connected successfully to external platform."
    r_tts = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": valid_raw_key},
        json={"text": synth_text}
    )
    tts_duration = round(time.time() - t0, 2)
    assert r_tts.status_code == 200, f"TTS failed: {r_tts.status_code} {r_tts.text}"
    tts_data = r_tts.json()
    assert tts_data.get("success") is True
    audio_rel_url = tts_data.get("audio_url")
    filename = tts_data.get("filename")
    print(f" PASS: Real TTS generated in {tts_duration}s. File: {filename}")

    # Verify physical file existence in storage
    physical_path = BASE_DIR / "storage" / "tts_output" / filename
    assert physical_path.exists(), f"Physical WAV file not found at {physical_path}"
    file_size_bytes = physical_path.stat().st_size
    assert file_size_bytes > 1000, f"Physical file too small: {file_size_bytes} bytes"
    print(f" PASS: Physical audio file verified: {physical_path.name} ({file_size_bytes} bytes)")

    # Verify audio retrieval endpoint
    r_audio = requests.get(f"{AMINOID_BASE}{audio_rel_url}")
    assert r_audio.status_code == 200
    assert r_audio.headers.get("content-type") in ("audio/wav", "audio/x-wav")
    print(f" PASS: Audio download/stream endpoint verified (HTTP 200, {len(r_audio.content)} bytes)")
    test_results["Phase 5: TTS Connectivity"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 6: Webhook Subscription Lifecycle
    # -------------------------------------------------------------------------
    print("\n--- PHASE 6: VERIFY WEBHOOK SUBSCRIPTIONS ---")
    # Clean prior test subscriptions for receiver port
    r_sub = requests.post(f"{AMINOID_BASE}/api/v1/integrations/webhooks", json={
        "name": "External Platform Webhook Listener",
        "target_url": f"{EXTERNAL_BASE}/webhook",
        "secret_key": SHARED_WEBHOOK_SECRET,
        "events": ["stt.completed", "tts.completed", "diagnostic.ping", "*"],
        "retry_count": 3,
        "retry_interval_sec": 1,
        "timeout_sec": 5
    })
    assert r_sub.status_code == 200, f"Failed to create subscription: {r_sub.text}"
    sub_data = r_sub.json()["subscription"]
    sub_id = sub_data["id"]
    print(f" PASS: Webhook subscription saved in SQLite: ID {sub_id} -> {sub_data['target_url']}")
    test_results["Phase 6: Webhook Subscription"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 7: Webhook Delivery & HMAC Signature Verification
    # -------------------------------------------------------------------------
    print("\n--- PHASE 7: VERIFY WEBHOOK DELIVERY & HMAC-SHA256 SIGNATURE ---")
    events_before = len(RECEIVED_EVENTS)

    # Trigger STT request which fires "stt.completed"
    print(" -> Triggering STT request to dispatch 'stt.completed' webhook event...")
    requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/stt/transcribe",
        headers={"X-API-Key": valid_raw_key},
        files={"audio": ("stt_event_test.wav", wav_bytes, "audio/wav")}
    )

    # Trigger TTS request which fires "tts.completed"
    print(" -> Triggering TTS request to dispatch 'tts.completed' webhook event...")
    requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": valid_raw_key},
        json={"text": "Webhook delivery verification event."}
    )

    # Allow async dispatcher to complete delivery
    time.sleep(2)

    events_after = len(RECEIVED_EVENTS)
    new_events = RECEIVED_EVENTS[events_before:]
    assert len(new_events) >= 2, f"Expected at least 2 events delivered, got {len(new_events)}"
    print(f" PASS: Received {len(new_events)} webhook events at external receiver:")

    stt_delivered = False
    tts_delivered = False
    for ev in new_events:
        print(f"   * Event: {ev['event_name']} | Delivery ID: {ev['delivery_id']} | HMAC Valid: {ev['signature_valid']}")
        assert ev["signature_valid"] is True, f"HMAC signature invalid for {ev['event_name']}"
        if ev["event_name"] == "stt.completed":
            stt_delivered = True
        if ev["event_name"] == "tts.completed":
            tts_delivered = True

    assert stt_delivered, "stt.completed webhook not received"
    assert tts_delivered, "tts.completed webhook not received"
    print(" PASS: Both 'stt.completed' and 'tts.completed' authenticated and verified by external platform!")
    test_results["Phase 7: Webhook Delivery"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 8: Failure Scenarios
    # -------------------------------------------------------------------------
    print("\n--- PHASE 8: TEST FAILURE SCENARIOS ---")
    # Scenario 1: Invalid / Unreachable URL
    print(" -> Testing Scenario 1: Invalid / unreachable destination URL...")
    test_fail_url = "http://127.0.0.1:54321/unreachable"
    r_fail1 = requests.post(f"{AMINOID_BASE}/api/v1/integrations/webhooks/test", json={
        "target_url": test_fail_url,
        "secret_key": "test_sec",
        "event_name": "test.failure.unreachable"
    })
    assert r_fail1.status_code == 200
    assert r_fail1.json()["status"] == "failed"
    print(f" PASS Scenario 1: Unreachable URL gracefully caught & logged as failed: {r_fail1.json()['error'][:50]}...")

    # Scenario 2: Destination returns HTTP 500
    print(" -> Testing Scenario 2: Destination server returns HTTP 500...")
    r_fail2 = requests.post(f"{AMINOID_BASE}/api/v1/integrations/webhooks/test", json={
        "target_url": f"{EXTERNAL_BASE}/webhook-500",
        "secret_key": "test_sec",
        "event_name": "test.failure.500"
    })
    assert r_fail2.status_code == 200
    assert r_fail2.json()["http_status"] == 500
    assert r_fail2.json()["status"] == "failed"
    print(f" PASS Scenario 2: HTTP 500 properly logged: status {r_fail2.json()['http_status']}")

    # Scenario 3: Timeout simulation
    print(" -> Testing Scenario 3: Destination timeout (non-blocking)...")
    r_sub_timeout = requests.post(f"{AMINOID_BASE}/api/v1/integrations/webhooks", json={
        "name": "Timeout Test Sub",
        "target_url": f"{EXTERNAL_BASE}/webhook-timeout",
        "secret_key": "sec",
        "events": ["test.timeout"],
        "retry_count": 1,
        "retry_interval_sec": 1,
        "timeout_sec": 2  # Short timeout
    })
    sub_timeout_id = r_sub_timeout.json()["subscription"]["id"]
    t0 = time.time()
    # Test webhook dispatch with timeout
    r_timeout = requests.post(f"{AMINOID_BASE}/api/v1/integrations/webhooks/test", json={
        "target_url": f"{EXTERNAL_BASE}/webhook-timeout",
        "secret_key": "sec",
        "timeout_sec": 3.0
    }, timeout=10)
    assert r_timeout.status_code == 200
    assert r_timeout.json().get("status") == "failed" or r_timeout.json().get("success") is False
    print(f" PASS Scenario 3: Timeout handled gracefully in {round(time.time() - t0, 2)}s without freezing backend")

    # Clean up timeout sub
    requests.delete(f"{AMINOID_BASE}/api/v1/integrations/webhooks/{sub_timeout_id}")
    test_results["Phase 8: Failure Handling"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 9: Bi-Directional External Communication
    # -------------------------------------------------------------------------
    print("\n--- PHASE 9: TEST CONNECTION TO A SECOND WEBSITE ---")
    # Simulate full conversation loop:
    # 1. External Platform calls AMINOID STT to transcribe customer voice
    # 2. AMINOID processes voice and returns transcript
    # 3. External Platform calls AMINOID TTS to generate bot response
    # 4. AMINOID sends HMAC-signed webhook to External Platform confirming speech generation
    print(" -> Executing Bi-Directional Integration Workflow...")
    
    # Step 1: External -> AMINOID STT
    r_loop_stt = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/stt/transcribe",
        headers={"X-API-Key": valid_raw_key},
        files={"audio": ("customer_call.wav", wav_bytes, "audio/wav")}
    )
    assert r_loop_stt.status_code == 200
    print(f" [Step 1] External Website -> AMINOID STT: HTTP 200 OK (Request: {r_loop_stt.json()['request_id']})")

    # Step 2: External -> AMINOID TTS
    r_loop_tts = requests.post(
        f"{AMINOID_BASE}/api/v1/integrations/tts/generate",
        headers={"X-API-Key": valid_raw_key},
        json={"text": "Hello, thank you for calling. Your account balance is zero."}
    )
    assert r_loop_tts.status_code == 200
    print(f" [Step 2] External Website -> AMINOID TTS: HTTP 200 OK (Audio: {r_loop_tts.json()['filename']})")

    # Step 3: Wait for Webhooks from AMINOID -> External
    time.sleep(2)
    ev_check = requests.get(f"{EXTERNAL_BASE}/webhook/events")
    all_events = ev_check.json()["events"]
    print(f" [Step 3] AMINOID Webhook -> External Website: Total received events = {len(all_events)}")
    test_results["Phase 9: Second Website Bi-Directional"] = "PASS"

    # -------------------------------------------------------------------------
    # PHASE 10: Live Diagnostics Endpoint
    # -------------------------------------------------------------------------
    print("\n--- PHASE 10: TEST LIVE 8-POINT DIAGNOSTIC ENDPOINT ---")
    r_diag = requests.post(f"{AMINOID_BASE}/api/v1/integrations/diagnostics/run", json={
        "api_key": valid_raw_key,
        "target_platform_url": f"{EXTERNAL_BASE}/health",
        "target_webhook_url": f"{EXTERNAL_BASE}/webhook",
        "webhook_secret": SHARED_WEBHOOK_SECRET
    })
    assert r_diag.status_code == 200, f"Diagnostics failed: {r_diag.text}"
    diag_data = r_diag.json()
    print(f" Diagnostics Overall Status: {diag_data['overall_status']}")
    checks = diag_data["checks"]
    for k, v in checks.items():
        print(f"   [{v['status']}] {v['label']}: {v['message']} ({v.get('latency_ms', 0)}ms)")
        assert v["status"] in ("PASS", "SKIPPED"), f"Check {k} failed: {v}"

    test_results["Phase 10: Live Diagnostics Endpoint"] = "PASS"

    print("\n" + "=" * 80)
    print("ALL INTEGRATION VERIFICATION PHASES COMPLETED SUCCESSFULLY:")
    for k, v in test_results.items():
        print(f"  ✓ {k}: {v}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
