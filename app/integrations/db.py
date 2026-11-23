"""
SQLite Database Storage Layer for AMINOID Speech Integrations.
Manages connections with WAL mode, initializes tables, and provides CRUD operations
for integrations, API keys, webhook subscriptions, delivery logs, and API request logs.
"""

import sqlite3
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("integrations.db")

DB_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "integrations.db"

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured with WAL mode and row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db() -> None:
    """Initializes tables and indexes if they do not exist."""
    conn = get_connection()
    try:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS integrations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                platform_type TEXT NOT NULL,
                description TEXT,
                base_url TEXT,
                webhook_url TEXT,
                auth_method TEXT DEFAULT 'none',
                api_key_encrypted TEXT,
                secret_key_encrypted TEXT,
                custom_headers TEXT DEFAULT '{}',
                timeout_sec INTEGER DEFAULT 30,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                permissions TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                expires_at TEXT
            );

            CREATE TABLE IF NOT EXISTS webhook_subscriptions (
                id TEXT PRIMARY KEY,
                integration_id TEXT,
                name TEXT NOT NULL,
                target_url TEXT NOT NULL,
                events TEXT NOT NULL,
                secret_key TEXT NOT NULL,
                custom_headers TEXT DEFAULT '{}',
                retry_count INTEGER DEFAULT 3,
                retry_interval_sec INTEGER DEFAULT 5,
                timeout_sec INTEGER DEFAULT 10,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (integration_id) REFERENCES integrations(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id TEXT PRIMARY KEY,
                subscription_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                target_url TEXT NOT NULL,
                payload TEXT NOT NULL,
                http_status INTEGER,
                response_body TEXT,
                duration_ms REAL,
                attempt_count INTEGER DEFAULT 1,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_request_logs (
                id TEXT PRIMARY KEY,
                api_key_id TEXT,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                client_ip TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_integrations_active ON integrations(is_active);
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_webhook_subs_active ON webhook_subscriptions(is_active);
            CREATE INDEX IF NOT EXISTS idx_deliveries_created ON webhook_deliveries(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_request_logs(created_at DESC);
            """)
        logger.info(f"Integrations SQLite database initialized at {DB_PATH}")
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Integrations CRUD
# -----------------------------------------------------------------------------
def create_integration(
    name: str,
    platform_type: str,
    description: Optional[str] = None,
    base_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
    auth_method: str = "none",
    api_key_encrypted: Optional[str] = None,
    secret_key_encrypted: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout_sec: int = 30
) -> Dict[str, Any]:
    intg_id = f"intg_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    headers_str = json.dumps(custom_headers or {})
    
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO integrations (
                    id, name, platform_type, description, base_url, webhook_url,
                    auth_method, api_key_encrypted, secret_key_encrypted,
                    custom_headers, timeout_sec, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    intg_id, name, platform_type, description, base_url, webhook_url,
                    auth_method, api_key_encrypted, secret_key_encrypted,
                    headers_str, timeout_sec, now, now
                )
            )
        return get_integration(intg_id)
    finally:
        conn.close()

def get_integration(intg_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM integrations WHERE id = ?", (intg_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
        # Mask sensitive keys in output
        d["has_api_key"] = bool(d.get("api_key_encrypted"))
        d["has_secret_key"] = bool(d.get("secret_key_encrypted"))
        d.pop("api_key_encrypted", None)
        d.pop("secret_key_encrypted", None)
        return d
    finally:
        conn.close()

def list_integrations(active_only: bool = False) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        query = "SELECT * FROM integrations"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
            d["has_api_key"] = bool(d.get("api_key_encrypted"))
            d["has_secret_key"] = bool(d.get("secret_key_encrypted"))
            d.pop("api_key_encrypted", None)
            d.pop("secret_key_encrypted", None)
            items.append(d)
        return items
    finally:
        conn.close()

def update_integration(intg_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        allowed = [
            "name", "platform_type", "description", "base_url", "webhook_url",
            "auth_method", "api_key_encrypted", "secret_key_encrypted",
            "custom_headers", "timeout_sec", "is_active"
        ]
        updates = []
        params = []
        for k in allowed:
            if k in fields:
                val = fields[k]
                if k == "custom_headers" and isinstance(val, dict):
                    val = json.dumps(val)
                updates.append(f"{k} = ?")
                params.append(val)
        if not updates:
            return get_integration(intg_id)
        updates.append("updated_at = ?")
        params.append(now)
        params.append(intg_id)

        with conn:
            conn.execute(f"UPDATE integrations SET {', '.join(updates)} WHERE id = ?", params)
        return get_integration(intg_id)
    finally:
        conn.close()

def delete_integration(intg_id: str) -> bool:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("DELETE FROM integrations WHERE id = ?", (intg_id,))
            return cur.rowcount > 0
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# API Keys CRUD
# -----------------------------------------------------------------------------
def create_api_key_record(
    key_id: str,
    name: str,
    key_prefix: str,
    key_hash: str,
    permissions: List[str],
    expires_at: Optional[str] = None
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    perms_str = json.dumps(permissions)
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    id, name, key_prefix, key_hash, permissions, is_active,
                    created_at, last_used_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, NULL, ?)
                """,
                (key_id, name, key_prefix, key_hash, perms_str, now, expires_at)
            )
        return get_api_key_record(key_id)
    finally:
        conn.close()

def get_api_key_record(key_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["permissions"] = json.loads(d.get("permissions") or "[]")
        d.pop("key_hash", None)
        return d
    finally:
        conn.close()

def get_api_key_by_hash(key_hash: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1", (key_hash,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["permissions"] = json.loads(d.get("permissions") or "[]")
        return d
    finally:
        conn.close()

def update_api_key_last_used(key_id: str) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id))
    except Exception as e:
        logger.warning(f"Failed to update api_key last_used_at: {e}")
    finally:
        conn.close()

def list_api_keys() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["permissions"] = json.loads(d.get("permissions") or "[]")
            d.pop("key_hash", None)
            items.append(d)
        return items
    finally:
        conn.close()

def toggle_api_key(key_id: str, is_active: bool) -> bool:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("UPDATE api_keys SET is_active = ? WHERE id = ?", (1 if is_active else 0, key_id))
            return cur.rowcount > 0
    finally:
        conn.close()

def revoke_api_key(key_id: str) -> bool:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            return cur.rowcount > 0
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Webhook Subscriptions CRUD
# -----------------------------------------------------------------------------
def create_webhook_subscription(
    name: str,
    target_url: str,
    events: List[str],
    secret_key: str,
    integration_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    retry_count: int = 3,
    retry_interval_sec: int = 5,
    timeout_sec: int = 10
) -> Dict[str, Any]:
    sub_id = f"sub_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    events_str = json.dumps(events)
    headers_str = json.dumps(custom_headers or {})

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO webhook_subscriptions (
                    id, integration_id, name, target_url, events, secret_key,
                    custom_headers, retry_count, retry_interval_sec, timeout_sec,
                    is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    sub_id, integration_id, name, target_url, events_str, secret_key,
                    headers_str, retry_count, retry_interval_sec, timeout_sec, now, now
                )
            )
        return get_webhook_subscription(sub_id)
    finally:
        conn.close()

def get_webhook_subscription(sub_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM webhook_subscriptions WHERE id = ?", (sub_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["events"] = json.loads(d.get("events") or "[]")
        d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
        # Mask secret key in general reading
        d["secret_key_preview"] = d["secret_key"][:6] + "..." if d.get("secret_key") else ""
        return d
    finally:
        conn.close()

def list_webhook_subscriptions(active_only: bool = False) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        query = "SELECT * FROM webhook_subscriptions"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["events"] = json.loads(d.get("events") or "[]")
            d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
            d["secret_key_preview"] = d["secret_key"][:6] + "..." if d.get("secret_key") else ""
            items.append(d)
        return items
    finally:
        conn.close()

def get_subscriptions_for_event(event_name: str) -> List[Dict[str, Any]]:
    """Retrieves all active subscriptions that subscribed to event_name or wildcard '*'."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM webhook_subscriptions WHERE is_active = 1").fetchall()
        matching = []
        for r in rows:
            d = dict(r)
            evts = json.loads(d.get("events") or "[]")
            if "*" in evts or event_name in evts:
                d["events"] = evts
                d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
                matching.append(d)
        return matching
    finally:
        conn.close()

def update_webhook_subscription(sub_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        allowed = [
            "name", "target_url", "events", "secret_key", "custom_headers",
            "retry_count", "retry_interval_sec", "timeout_sec", "is_active"
        ]
        updates = []
        params = []
        for k in allowed:
            if k in fields:
                val = fields[k]
                if k in ("events", "custom_headers") and not isinstance(val, str):
                    val = json.dumps(val)
                updates.append(f"{k} = ?")
                params.append(val)
        if not updates:
            return get_webhook_subscription(sub_id)
        updates.append("updated_at = ?")
        params.append(now)
        params.append(sub_id)

        with conn:
            conn.execute(f"UPDATE webhook_subscriptions SET {', '.join(updates)} WHERE id = ?", params)
        return get_webhook_subscription(sub_id)
    finally:
        conn.close()

def delete_webhook_subscription(sub_id: str) -> bool:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("DELETE FROM webhook_subscriptions WHERE id = ?", (sub_id,))
            return cur.rowcount > 0
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Webhook Deliveries Logging
# -----------------------------------------------------------------------------
def log_webhook_delivery(
    subscription_id: str,
    event_name: str,
    target_url: str,
    payload: Dict[str, Any],
    http_status: Optional[int],
    response_body: Optional[str],
    duration_ms: float,
    status: str,
    attempt_count: int = 1,
    error_message: Optional[str] = None
) -> str:
    del_id = f"del_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload_str = json.dumps(payload, ensure_ascii=False)

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO webhook_deliveries (
                    id, subscription_id, event_name, target_url, payload,
                    http_status, response_body, duration_ms, attempt_count,
                    status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    del_id, subscription_id, event_name, target_url, payload_str,
                    http_status, response_body[:2000] if response_body else None,
                    round(duration_ms, 2), attempt_count, status, error_message, now
                )
            )
        return del_id
    finally:
        conn.close()

def list_webhook_deliveries(limit: int = 50, event_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        query = "SELECT * FROM webhook_deliveries WHERE 1=1"
        params = []
        if event_name:
            query += " AND event_name = ?"
            params.append(event_name)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.get("payload") or "{}")
            except Exception:
                pass
            items.append(d)
        return items
    finally:
        conn.close()

def get_webhook_delivery(delivery_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM webhook_deliveries WHERE id = ?", (delivery_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["payload"] = json.loads(d.get("payload") or "{}")
        except Exception:
            pass
        return d
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# API Request Logs
# -----------------------------------------------------------------------------
def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    processing_time_ms: float,
    api_key_id: Optional[str] = None,
    client_ip: Optional[str] = None
) -> None:
    req_id = f"req_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO api_request_logs (
                    id, api_key_id, endpoint, method, status_code,
                    processing_time_ms, client_ip, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (req_id, api_key_id, endpoint, method, status_code, round(processing_time_ms, 2), client_ip, now)
            )
    except Exception as e:
        logger.warning(f"Failed to log API request: {e}")
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Aggregated Statistics
# -----------------------------------------------------------------------------
def get_integration_stats() -> Dict[str, Any]:
    """Computes real-time statistics across API requests and webhook deliveries."""
    conn = get_connection()
    try:
        today_date = time.strftime("%Y-%m-%d")
        
        # 1. Total requests today
        row_today = conn.execute(
            "SELECT COUNT(*) as total, AVG(processing_time_ms) as avg_time FROM api_request_logs WHERE created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()
        total_today = row_today["total"] if row_today else 0
        avg_processing_time = round(row_today["avg_time"] or 0.0, 2) if row_today else 0.0

        # 2. STT vs TTS request counts
        stt_today = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE endpoint LIKE '%/stt/%' AND created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()["cnt"]

        tts_today = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE endpoint LIKE '%/tts/%' AND created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()["cnt"]

        # 3. Success vs Failure
        success_today = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE status_code >= 200 AND status_code < 400 AND created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()["cnt"]

        failed_today = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE status_code >= 400 AND created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()["cnt"]

        # 4. Active integrations & subscriptions
        active_integrations = conn.execute("SELECT COUNT(*) as cnt FROM integrations WHERE is_active = 1").fetchone()["cnt"]
        active_webhooks = conn.execute("SELECT COUNT(*) as cnt FROM webhook_subscriptions WHERE is_active = 1").fetchone()["cnt"]

        # 5. Webhook delivery statistics
        wh_stats = conn.execute(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM webhook_deliveries
            WHERE created_at LIKE ?
            """,
            (f"{today_date}%",)
        ).fetchone()

        return {
            "api_requests_today": total_today,
            "stt_requests": stt_today,
            "tts_requests": tts_today,
            "successful_requests": success_today,
            "failed_requests": failed_today,
            "avg_processing_time_ms": avg_processing_time,
            "active_integrations": active_integrations,
            "active_webhook_subscriptions": active_webhooks,
            "webhook_deliveries_today": wh_stats["total"] if wh_stats else 0,
            "webhook_successful_today": wh_stats["successful"] if wh_stats and wh_stats["successful"] else 0,
            "webhook_failed_today": wh_stats["failed"] if wh_stats and wh_stats["failed"] else 0,
        }
    finally:
        conn.close()

# Auto-initialize database schema on module load
init_db()
