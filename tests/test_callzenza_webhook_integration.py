"""
Comprehensive End-to-End Verification for AMINOID ↔ CallZenza Webhook Integration.
Tests all live CallZenza APIs, HMAC-SHA256 signatures, Event Catalog, and delivery logs.
"""

import hmac
import hashlib
import json
import httpx

BASE_URL = "http://127.0.0.1:8080"

def test_callzenza_integration():
    client = httpx.Client(base_url=BASE_URL, timeout=15.0)

    print("=== Step 1: Query Dynamic Event Catalog ===")
    r = client.get("/api/v1/integrations/events")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    cat_data = r.json()
    assert cat_data.get("success") is True
    assert len(cat_data.get("categories", [])) == 7, "Expected 7 categories"
    assert cat_data.get("total_events", 0) >= 29, "Expected at least 29 events"
    print(f"✓ Event Catalog OK: {len(cat_data['categories'])} categories, {cat_data['total_events']} events")

    print("\n=== Step 2: Query Initial CallZenza Status ===")
    r = client.get("/api/v1/integrations/callzenza/status")
    assert r.status_code == 200
    cz_status = r.json()
    assert cz_status.get("success") is True
    print(f"✓ CallZenza Status OK (Initially connected={cz_status.get('connected')})")

    print("\n=== Step 3: Connect CallZenza AI Receptionist ===")
    connect_payload = {
        "destination_url": f"{BASE_URL}/webhook",
        "secret_key": "whsec_callzenza_prod_secret_9988",
        "auth_token": "cz_bearer_token_abc123",
        "base_url": BASE_URL,
        "description": "CallZenza AI Receptionist production webhook",
        "events": [
            "stt.transcription.completed",
            "tts.generation.completed",
            "tts.audio.generated",
            "callzenza.session.started",
            "callzenza.audio.processed",
            "callzenza.call.ended",
            "voice.clone.completed"
        ]
    }
    r = client.post("/api/v1/integrations/callzenza/connect", json=connect_payload)
    assert r.status_code == 200
    res = r.json()
    assert res.get("success") is True
    assert res.get("integration", {}).get("name") == "CallZenza AI Receptionist"
    assert res.get("subscription", {}).get("target_url") == f"{BASE_URL}/webhook"
    print(f"✓ CallZenza Connected: Subscription ID = {res['subscription']['id']}")

    print("\n=== Step 4: Verify CallZenza Status After Connect ===")
    r = client.get("/api/v1/integrations/callzenza/status")
    assert r.status_code == 200
    cz_status = r.json()
    assert cz_status.get("connected") is True
    assert cz_status.get("subscription") is not None
    print(f"✓ Verified connected status: {cz_status.get('connected')}")

    print("\n=== Step 5: Send Test Webhook to CallZenza with HMAC Signature ===")
    test_payload = {
        "event_name": "callzenza.session.started",
        "sample_payload": {
            "session_id": "cz_test_live_001",
            "caller": "+1-800-CALLZENZA",
            "agent": "CallZenza Virtual Agent",
            "source": "AMINOID-Offline-Engine"
        }
    }
    r = client.post("/api/v1/integrations/callzenza/test", json=test_payload)
    assert r.status_code == 200
    test_res = r.json()
    assert test_res.get("status") == "success"
    assert test_res.get("http_status") == 200
    assert test_res.get("signature_verified") is True
    print(f"✓ Test Webhook Delivered: HTTP {test_res.get('http_status')} in {test_res.get('duration_ms')}ms")

    print("\n=== Step 6: Simulate CallZenza Inbound Call Session ===")
    session_payload = {
        "session_id": "cz_sess_auto_101",
        "caller_number": "+1 (555) 019-2834",
        "agent_name": "CallZenza AI Receptionist",
        "initial_prompt": "Hello! Welcome to our automated medical scheduling service."
    }
    r = client.post("/api/v1/integrations/callzenza/session", json=session_payload)
    assert r.status_code == 200
    sess_res = r.json()
    assert sess_res.get("success") is True
    assert sess_res.get("session", {}).get("session_id") == "cz_sess_auto_101"
    print(f"✓ Inbound Call Session Simulated: ID = {sess_res['session']['session_id']}")

    print("\n=== Step 7: Inspect Delivery Logs and Request Headers ===")
    r = client.get("/api/v1/integrations/webhooks/deliveries?limit=10")
    assert r.status_code == 200
    delivs = r.json().get("deliveries", [])
    assert len(delivs) > 0
    latest = delivs[0]
    headers = latest.get("request_headers", {})
    assert "X-Aminoid-Signature" in headers, "Expected X-Aminoid-Signature in headers"
    assert "X-Aminoid-Event" in headers, "Expected X-Aminoid-Event in headers"
    assert "X-Aminoid-Delivery-Id" in headers, "Expected X-Aminoid-Delivery-Id in headers"
    print(f"✓ Delivery Headers Verified: Signature = {headers.get('X-Aminoid-Signature')[:25]}...")

    print("\n=== ALL CALLZENZA INTEGRATION TESTS PASSED 100% ===")

if __name__ == "__main__":
    test_callzenza_integration()
