"""
External Platform Receiver & Bi-Directional Connectivity Verification Server.
Simulates a genuine external website, AI voice agent, or CRM platform running on port 9090.
Provides:
1. Real Webhook Receiver Endpoint (POST /webhook) with HMAC-SHA256 signature verification.
2. Error Simulation Endpoint (POST /webhook-500) for testing failure recovery.
3. Timeout Simulation Endpoint (POST /webhook-timeout) for testing async non-blocking execution.
4. Events Log Endpoint (GET /webhook/events) for inspecting received deliveries.
5. Bi-directional client caller (calls AMINOID STT/TTS API with API key).
"""

import sys
import os
import json
import time
import hmac
import hashlib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any
import requests

RECEIVED_EVENTS: List[Dict[str, Any]] = []
SERVER_PORT = 9090
SHARED_WEBHOOK_SECRET = "whsec_external_platform_test_secret_99"

def verify_hmac_signature(secret: str, timestamp: str, raw_payload: str, signature_header: str) -> bool:
    """Verifies X-Aminoid-Signature matches expected HMAC-SHA256."""
    try:
        content = f"{timestamp}.{raw_payload}".encode("utf-8")
        expected_digest = hmac.new(secret.encode("utf-8"), content, hashlib.sha256).hexdigest()
        expected_sig = f"sha256={expected_digest}"
        return hmac.compare_digest(expected_sig, signature_header)
    except Exception:
        return False

class ExternalPlatformHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet standard HTTP server logs during automated test runs
        return

    def do_GET(self):
        if self.path == "/webhook/events":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "count": len(RECEIVED_EVENTS),
                "events": RECEIVED_EVENTS
            }).encode("utf-8"))
        elif self.path in ("/webhook", "/webhook-500", "/webhook-timeout"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ready",
                "message": "Webhook endpoint active (POST required for deliveries)"
            }).encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "online",
                "service": "External Platform Receiver",
                "port": SERVER_PORT
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""

        if self.path == "/webhook":
            event_name = self.headers.get("X-Aminoid-Event", "unknown")
            delivery_id = self.headers.get("X-Aminoid-Delivery-Id", "unknown")
            timestamp = self.headers.get("X-Aminoid-Timestamp", "")
            signature = self.headers.get("X-Aminoid-Signature", "")

            # Verify HMAC-SHA256 signature
            sig_valid = verify_hmac_signature(SHARED_WEBHOOK_SECRET, timestamp, raw_body, signature)

            parsed_payload = {}
            try:
                parsed_payload = json.loads(raw_body)
            except Exception:
                pass

            record = {
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_name": event_name,
                "delivery_id": delivery_id,
                "timestamp_header": timestamp,
                "signature_header": signature,
                "signature_valid": sig_valid,
                "payload": parsed_payload,
                "headers": dict(self.headers)
            }
            RECEIVED_EVENTS.append(record)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "received": True,
                "event": event_name,
                "delivery_id": delivery_id,
                "signature_valid": sig_valid
            }).encode("utf-8"))

        elif self.path == "/webhook-500":
            # Simulate destination internal server error for Phase 8 Scenario 2
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Simulated external platform internal error",
                "status": 500
            }).encode("utf-8"))

        elif self.path == "/webhook-timeout":
            # Simulate slow / non-responsive destination for Phase 8 Scenario 3
            time.sleep(6)
            self.send_response(200)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

def start_external_receiver_server(port: int = SERVER_PORT) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), ExternalPlatformHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

if __name__ == "__main__":
    server = start_external_receiver_server(SERVER_PORT)
    print(f"External platform test receiver listening on http://127.0.0.1:{SERVER_PORT}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
