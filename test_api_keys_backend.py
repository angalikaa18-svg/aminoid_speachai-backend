"""
Comprehensive Backend Test Suite for AMINOID API Key Management & Developer Integration
"""
import time
from fastapi.testclient import TestClient
from app.main import app
from app.integrations.auth import generate_api_key, hash_api_key, _rate_limiter
from app.integrations.db import (
    get_api_key_stats,
    get_api_key_analytics,
    list_api_keys_paginated,
    list_api_request_logs_paginated,
    get_api_key_record,
    toggle_api_key,
    revoke_api_key
)

def run_tests():
    client = TestClient(app)
    print("=" * 60)
    print("STARTING AMINOID API KEY & GATEWAY TEST SUITE")
    print("=" * 60)

    # 1. Generate a Production API Key
    raw_key, record = generate_api_key(
        name="CallZenza Telephony Agent",
        client_name="CallZenza AI Receptionist",
        environment="production",
        permissions=["stt.transcribe", "tts.generate", "tts.voices.read", "audio.read", "models.read"],
        rate_limit_rpm=10,
        expires_in_days=30
    )
    print(f"[PASS] Key Generated: Prefix={record['key_prefix']}, ID={record['id']}")
    assert raw_key.startswith("aminoid_live_"), f"Prefix wrong: {raw_key}"
    assert "key_hash" not in record, "Raw hash must not be exposed in public record"
    assert record["status"] == "active"

    # 2. Verify Stats Endpoint
    res = client.get("/api/v1/integrations/api-keys/stats")
    assert res.status_code == 200, f"Stats failed: {res.text}"
    stats_data = res.json()["stats"]
    print(f"[PASS] API Key Stats: Total={stats_data['total_keys']}, Active={stats_data['active_keys']}")
    assert stats_data["total_keys"] >= 1

    # 3. Test Valid API Key Authentication on Voices endpoint
    res = client.get(
        "/api/v1/integrations/tts/voices",
        headers={"Authorization": f"Bearer {raw_key}"}
    )
    assert res.status_code == 200, f"Valid auth failed: {res.status_code} {res.text}"
    print(f"[PASS] Valid Bearer Key Authenticated (Voices count: {res.json()['count']})")

    # 4. Test Missing Auth when strict header is present
    res = client.get(
        "/api/v1/integrations/tts/voices",
        headers={"X-Require-Auth": "strict"}
    )
    assert res.status_code == 401, f"Expected 401 for missing key, got {res.status_code}"
    print("[PASS] Missing API Key rejected with 401 Unauthorized")

    # 5. Test Invalid API Key
    res = client.get(
        "/api/v1/integrations/tts/voices",
        headers={"Authorization": "Bearer aminoid_live_invalid_garbage_key_12345"}
    )
    assert res.status_code == 401, f"Expected 401 for invalid key, got {res.status_code}"
    print("[PASS] Invalid API Key rejected with 401 Unauthorized")

    # 6. Test Permission Restriction
    # Generate key with ONLY tts.voices.read permission
    limited_raw_key, limited_rec = generate_api_key(
        name="Voice Catalog Only Key",
        permissions=["tts.voices.read"]
    )
    # Attempting to call STT models endpoint (requires stt.models.read)
    res = client.get(
        "/api/v1/integrations/stt/models",
        headers={"Authorization": f"Bearer {limited_raw_key}"}
    )
    assert res.status_code == 403, f"Expected 403 Forbidden for missing permission, got {res.status_code}"
    print(f"[PASS] Permission Boundary Enforced: 403 Forbidden ({res.json()['detail']})")

    # 7. Test Deactivated Key
    toggle_api_key(limited_rec["id"], False)
    res = client.get(
        "/api/v1/integrations/tts/voices",
        headers={"Authorization": f"Bearer {limited_raw_key}"}
    )
    assert res.status_code in (401, 403), f"Expected 401/403 for deactivated key, got {res.status_code}"
    print("[PASS] Deactivated Key rejected with 401/403")

    # 8. Test Revoked Key
    rev_raw_key, rev_rec = generate_api_key(name="Key to Revoke")
    revoke_api_key(rev_rec["id"])
    res = client.get(
        "/api/v1/integrations/tts/voices",
        headers={"Authorization": f"Bearer {rev_raw_key}"}
    )
    assert res.status_code == 401, f"Expected 401 for revoked key, got {res.status_code}"
    print("[PASS] Revoked Key rejected with 401 Unauthorized")

    # 9. Test Rate Limiting
    # Generate key with rate_limit_rpm=3
    rate_raw_key, rate_rec = generate_api_key(
        name="Rate Limited Key",
        permissions=["tts.voices.read"],
        rate_limit_rpm=3
    )
    for i in range(3):
        res = client.get("/api/v1/integrations/tts/voices", headers={"Authorization": f"Bearer {rate_raw_key}"})
        assert res.status_code == 200, f"Attempt {i+1} failed"
    # 4th request must trigger 429 Too Many Requests
    res4 = client.get("/api/v1/integrations/tts/voices", headers={"Authorization": f"Bearer {rate_raw_key}"})
    assert res4.status_code == 429, f"Expected 429 Too Many Requests, got {res4.status_code}"
    print(f"[PASS] Rate Limiter Enforced: 429 Too Many Requests ({res4.json()['detail']})")

    # 10. Test Connection Diagnostic API (Comprehensive Permutations - NEVER 422)
    # A: Standard JSON { "api_key": raw_key }
    res = client.post("/api/v1/integrations/api-keys/test-connection", json={"api_key": raw_key})
    assert res.status_code == 200, f"Test connection standard failed: {res.text}"
    conn_data = res.json()
    assert conn_data["authenticated"] is True
    assert conn_data["status"] == "PASS"

    # B: Empty JSON body {} (Must return 200, NEVER 422)
    res_empty = client.post("/api/v1/integrations/api-keys/test-connection", json={})
    assert res_empty.status_code == 200, f"Empty JSON returned {res_empty.status_code}: {res_empty.text}"
    print("[PASS] Empty JSON body {} returned 200 OK (no 422)")

    # C: No body at all
    res_nobody = client.post("/api/v1/integrations/api-keys/test-connection")
    assert res_nobody.status_code == 200, f"No body returned {res_nobody.status_code}: {res_nobody.text}"
    print("[PASS] No request body returned 200 OK (no 422)")

    # D: CamelCase alias { "apiKey": raw_key }
    res_camel = client.post("/api/v1/integrations/api-keys/test-connection", json={"apiKey": raw_key})
    assert res_camel.status_code == 200, f"CamelCase returned {res_camel.status_code}"
    assert res_camel.json()["authenticated"] is True
    print("[PASS] CamelCase alias { 'apiKey': ... } supported")

    # E: Query Param via GET
    res_get = client.get(f"/api/v1/integrations/api-keys/test-connection?api_key={raw_key}")
    assert res_get.status_code == 200, f"GET returned {res_get.status_code}"
    assert res_get.json()["authenticated"] is True
    print("[PASS] GET with query parameter supported")

    # F: Test with Prefix (like UI table button)
    res_pfx = client.post("/api/v1/integrations/api-keys/test-connection", json={"api_key": record["key_prefix"]})
    assert res_pfx.status_code == 200, f"Prefix test returned {res_pfx.status_code}"
    assert res_pfx.json()["authenticated"] is True
    print(f"[PASS] Connection Test Succeeded via Prefix: Latency={conn_data['latency_ms']}ms, Services={list(conn_data['services'].keys())}")

    # 11. Test Paginated Logs Endpoint
    res = client.get("/api/v1/integrations/api-keys/logs?limit=10")
    assert res.status_code == 200, f"Logs query failed: {res.text}"
    logs_data = res.json()["logs"]
    print(f"[PASS] Request Logs Queried: Total={logs_data['total']} logs recorded in SQLite")

    # 12. Test Analytics Endpoint
    res = client.get("/api/v1/integrations/api-keys/analytics")
    assert res.status_code == 200, f"Analytics failed: {res.text}"
    an_data = res.json()["analytics"]
    print(f"[PASS] Analytics Computed: Endpoints tracked={len(an_data['by_endpoint'])}, Keys tracked={len(an_data['by_key'])}")

    print("=" * 60)
    print("ALL 12 API KEY & INTEGRATION TESTS PASSED 100%!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
