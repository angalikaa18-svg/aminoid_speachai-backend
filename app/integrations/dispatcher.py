"""
Centralized Asynchronous Webhook Dispatcher for AMINOID Speech Integrations.
Computes HMAC-SHA256 signatures, manages asynchronous HTTP delivery with retry backoff,
logs delivery results to SQLite, and ensures speech engines are never blocked.
"""

import asyncio
import hmac
import hashlib
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
import httpx

from .db import (
    get_subscriptions_for_event,
    log_webhook_delivery,
    get_webhook_subscription,
    get_webhook_delivery
)

logger = logging.getLogger("integrations.dispatcher")

def compute_hmac_signature(secret_key: str, timestamp: str, payload_str: str) -> str:
    """
    Computes an HMAC-SHA256 signature over timestamp and raw payload string.
    Format: sha256=<hex_digest>
    """
    signed_content = f"{timestamp}.{payload_str}".encode("utf-8")
    secret_bytes = secret_key.strip().encode("utf-8")
    digest = hmac.new(secret_bytes, signed_content, hashlib.sha256).hexdigest()
    return f"sha256={digest}"

async def _deliver_to_subscriber(
    subscription: Dict[str, Any],
    event_name: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Delivers a single webhook event to a subscriber with retries and exponential backoff.
    Logs every attempt to the SQLite database.
    """
    sub_id = subscription["id"]
    target_url = subscription["target_url"]
    secret_key = subscription.get("secret_key", "default_secret")
    max_retries = subscription.get("retry_count", 3)
    base_interval = subscription.get("retry_interval_sec", 5)
    timeout_sec = subscription.get("timeout_sec", 10)
    custom_headers = subscription.get("custom_headers", {})

    payload_str = json.dumps(payload, ensure_ascii=False)
    timestamp = payload.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    signature = compute_hmac_signature(secret_key, timestamp, payload_str)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AMINOID-Speech-Webhook-Dispatcher/1.0",
        "X-Aminoid-Event": event_name,
        "X-Aminoid-Delivery-Id": payload.get("event_id", str(uuid.uuid4())),
        "X-Aminoid-Timestamp": timestamp,
        "X-Aminoid-Signature": signature,
    }
    # Add subscriber custom headers
    if isinstance(custom_headers, dict):
        for k, v in custom_headers.items():
            headers[str(k)] = str(v)

    last_status = None
    last_response = None
    last_error = None
    last_duration = 0.0

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=float(timeout_sec), verify=False) as client:
                res = await client.post(target_url, content=payload_str, headers=headers)
                duration_ms = (time.time() - t0) * 1000.0
                last_duration = duration_ms
                last_status = res.status_code
                last_response = res.text[:2000]

                if res.is_success:
                    log_webhook_delivery(
                        subscription_id=sub_id,
                        event_name=event_name,
                        target_url=target_url,
                        payload=payload,
                        http_status=res.status_code,
                        response_body=last_response,
                        duration_ms=duration_ms,
                        status="success",
                        attempt_count=attempt
                    )
                    logger.info(f"Webhook [{event_name}] delivered to {target_url} in {duration_ms:.1f}ms (attempt {attempt})")
                    return {
                        "status": "success",
                        "http_status": res.status_code,
                        "duration_ms": duration_ms,
                        "attempt": attempt,
                        "response": last_response
                    }
                else:
                    last_error = f"HTTP {res.status_code}: {res.text[:200]}"
                    logger.warning(f"Webhook [{event_name}] delivery attempt {attempt} to {target_url} returned {res.status_code}")

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            last_duration = duration_ms
            last_error = str(e)
            logger.warning(f"Webhook [{event_name}] delivery attempt {attempt} to {target_url} error: {e}")

        # If not last attempt, wait with exponential backoff before retry
        if attempt < max_retries:
            backoff_sec = base_interval * (1.5 ** (attempt - 1))
            await asyncio.sleep(backoff_sec)

    # All attempts exhausted
    log_webhook_delivery(
        subscription_id=sub_id,
        event_name=event_name,
        target_url=target_url,
        payload=payload,
        http_status=last_status,
        response_body=last_response,
        duration_ms=last_duration,
        status="failed",
        attempt_count=max_retries,
        error_message=last_error
    )
    return {
        "status": "failed",
        "http_status": last_status,
        "duration_ms": last_duration,
        "attempt": max_retries,
        "error": last_error
    }

def dispatch_event(event_name: str, data: Dict[str, Any]) -> str:
    """
    Main non-blocking event dispatcher.
    Finds all active webhook subscriptions for the event and fires asynchronous delivery.
    Returns: event_id
    """
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload = {
        "event": event_name,
        "event_id": event_id,
        "timestamp": timestamp,
        "data": data
    }

    try:
        subscriptions = get_subscriptions_for_event(event_name)
        if not subscriptions:
            return event_id

        # Schedule async delivery in existing or background loop without blocking caller
        try:
            loop = asyncio.get_running_loop()
            for sub in subscriptions:
                loop.create_task(_deliver_to_subscriber(sub, event_name, payload))
        except RuntimeError:
            # If called from a non-async thread (e.g. sync worker thread)
            import threading
            def _run_bg():
                bg_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(bg_loop)
                tasks = [_deliver_to_subscriber(sub, event_name, payload) for sub in subscriptions]
                bg_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                bg_loop.close()

            threading.Thread(target=_run_bg, daemon=True).start()

    except Exception as e:
        logger.error(f"Error initiating event dispatch for {event_name}: {e}")

    return event_id

async def send_test_webhook(
    target_url: str,
    secret_key: str,
    event_name: str = "test.ping",
    sample_payload: Optional[Dict[str, Any]] = None,
    custom_headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Sends a real, synchronous test webhook request to a destination URL for live UI testing.
    Returns response details, status code, latency, and headers sent.
    """
    event_id = f"test_{uuid.uuid4().hex[:8]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload = {
        "event": event_name,
        "event_id": event_id,
        "timestamp": timestamp,
        "is_test": True,
        "data": sample_payload or {
            "message": "This is a real test webhook event from AMINOID Speech Integration Hub.",
            "source": "AMINOID-Speech-Engine",
            "version": "2.0.0"
        }
    }

    payload_str = json.dumps(payload, ensure_ascii=False)
    signature = compute_hmac_signature(secret_key or "test_secret", timestamp, payload_str)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AMINOID-Speech-Webhook-Tester/1.0",
        "X-Aminoid-Event": event_name,
        "X-Aminoid-Delivery-Id": event_id,
        "X-Aminoid-Timestamp": timestamp,
        "X-Aminoid-Signature": signature,
    }
    if custom_headers and isinstance(custom_headers, dict):
        for k, v in custom_headers.items():
            headers[str(k)] = str(v)

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            res = await client.post(target_url, content=payload_str, headers=headers)
            duration_ms = (time.time() - t0) * 1000.0
            return {
                "success": res.is_success,
                "http_status": res.status_code,
                "response_body": res.text[:2000],
                "duration_ms": round(duration_ms, 2),
                "event_id": event_id,
                "signature": signature,
                "headers_sent": {k: (v if "Signature" not in k else f"{v[:18]}...") for k, v in headers.items()},
                "payload": payload
            }
    except Exception as e:
        duration_ms = (time.time() - t0) * 1000.0
        return {
            "success": False,
            "http_status": None,
            "error": str(e),
            "duration_ms": round(duration_ms, 2),
            "event_id": event_id,
            "signature": signature,
            "headers_sent": headers,
            "payload": payload
        }
