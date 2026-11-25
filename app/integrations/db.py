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

            # Safe schema migrations for CallZenza & advanced webhooks
            for migration in [
                "ALTER TABLE webhook_subscriptions ADD COLUMN webhook_type TEXT DEFAULT 'outgoing'",
                "ALTER TABLE webhook_subscriptions ADD COLUMN description TEXT",
                "ALTER TABLE webhook_deliveries ADD COLUMN request_headers TEXT DEFAULT '{}'",
                "ALTER TABLE api_keys ADD COLUMN client_name TEXT DEFAULT ''",
                "ALTER TABLE api_keys ADD COLUMN client_id TEXT DEFAULT ''",
                "ALTER TABLE api_keys ADD COLUMN description TEXT DEFAULT ''",
                "ALTER TABLE api_keys ADD COLUMN environment TEXT DEFAULT 'production'",
                "ALTER TABLE api_keys ADD COLUMN status TEXT DEFAULT 'active'",
                "ALTER TABLE api_keys ADD COLUMN updated_at TEXT",
                "ALTER TABLE api_keys ADD COLUMN last_used_ip TEXT DEFAULT ''",
                "ALTER TABLE api_keys ADD COLUMN total_requests INTEGER DEFAULT 0",
                "ALTER TABLE api_keys ADD COLUMN successful_requests INTEGER DEFAULT 0",
                "ALTER TABLE api_keys ADD COLUMN failed_requests INTEGER DEFAULT 0",
                "ALTER TABLE api_keys ADD COLUMN rate_limit_rpm INTEGER DEFAULT 60",
                "ALTER TABLE api_keys ADD COLUMN rate_limit_rph INTEGER DEFAULT 1000",
                "ALTER TABLE api_keys ADD COLUMN rate_limit_rpd INTEGER DEFAULT 10000",
                "ALTER TABLE api_keys ADD COLUMN ip_whitelist TEXT DEFAULT '[]'",
                "ALTER TABLE api_keys ADD COLUMN owner_id TEXT DEFAULT 'admin'",
                "ALTER TABLE api_request_logs ADD COLUMN client_name TEXT DEFAULT ''",
                "ALTER TABLE api_request_logs ADD COLUMN error_message TEXT DEFAULT ''",
                "ALTER TABLE api_request_logs ADD COLUMN user_agent TEXT DEFAULT ''"
            ]:
                try:
                    conn.execute(migration)
                except Exception:
                    pass

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
def _normalize_api_key_record(d: Dict[str, Any], include_hash: bool = False) -> Dict[str, Any]:
    """Parses JSON fields, resolves dynamic expiration/revocation status, and strips key_hash."""
    if not d:
        return d
    
    # Parse permissions
    perms = d.get("permissions")
    if isinstance(perms, str):
        try:
            d["permissions"] = json.loads(perms or "[]")
        except Exception:
            d["permissions"] = []
    elif not isinstance(perms, list):
        d["permissions"] = []

    # Parse ip_whitelist
    ips = d.get("ip_whitelist")
    if isinstance(ips, str):
        try:
            d["ip_whitelist"] = json.loads(ips or "[]")
        except Exception:
            d["ip_whitelist"] = []
    elif not isinstance(ips, list):
        d["ip_whitelist"] = []

    # Defaults
    d["client_name"] = d.get("client_name") or "External Application"
    d["client_id"] = d.get("client_id") or ""
    d["description"] = d.get("description") or ""
    d["environment"] = d.get("environment") or "production"
    d["total_requests"] = int(d.get("total_requests") or 0)
    d["successful_requests"] = int(d.get("successful_requests") or 0)
    d["failed_requests"] = int(d.get("failed_requests") or 0)
    d["rate_limit_rpm"] = int(d.get("rate_limit_rpm") or 60)
    d["rate_limit_rph"] = int(d.get("rate_limit_rph") or 1000)
    d["rate_limit_rpd"] = int(d.get("rate_limit_rpd") or 10000)

    # Determine real-time status
    raw_status = (d.get("status") or "active").lower()
    is_active = int(d.get("is_active", 1))

    if raw_status == "revoked":
        d["status"] = "revoked"
        d["is_active"] = 0
    elif d.get("expires_at"):
        try:
            exp_str = d["expires_at"].replace("Z", "+00:00")
            from datetime import datetime, timezone
            if datetime.now(timezone.utc) > datetime.fromisoformat(exp_str):
                d["status"] = "expired"
                d["is_active"] = 0
            elif is_active == 0:
                d["status"] = "disabled"
                d["is_active"] = 0
            else:
                d["status"] = "active"
                d["is_active"] = 1
        except Exception:
            d["status"] = raw_status if is_active == 1 else "disabled"
    elif is_active == 0:
        d["status"] = "disabled"
    else:
        d["status"] = "active"

    if not include_hash:
        d.pop("key_hash", None)

    return d

def create_api_key_record(
    key_id: str,
    name: str,
    key_prefix: str,
    key_hash: str,
    permissions: List[str],
    client_name: str = "CallZenza AI Receptionist",
    client_id: Optional[str] = None,
    description: Optional[str] = None,
    environment: str = "production",
    expires_at: Optional[str] = None,
    rate_limit_rpm: int = 60,
    rate_limit_rph: int = 1000,
    rate_limit_rpd: int = 10000,
    ip_whitelist: Optional[List[str]] = None,
    owner_id: str = "admin"
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    perms_str = json.dumps(permissions)
    ips_str = json.dumps(ip_whitelist or [])
    
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    id, name, client_name, client_id, description, environment,
                    key_prefix, key_hash, permissions, status, is_active,
                    created_at, updated_at, last_used_at, expires_at,
                    total_requests, successful_requests, failed_requests,
                    rate_limit_rpm, rate_limit_rph, rate_limit_rpd,
                    ip_whitelist, owner_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, NULL, ?, 0, 0, 0, ?, ?, ?, ?, ?)
                """,
                (
                    key_id, name, client_name or "CallZenza AI Receptionist",
                    client_id or f"client_{uuid.uuid4().hex[:8]}", description or "",
                    environment or "production", key_prefix, key_hash, perms_str,
                    now, now, expires_at, rate_limit_rpm, rate_limit_rph,
                    rate_limit_rpd, ips_str, owner_id
                )
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
        return _normalize_api_key_record(dict(row), include_hash=False)
    finally:
        conn.close()

def get_api_key_by_hash(key_hash: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        if not row:
            return None
        return _normalize_api_key_record(dict(row), include_hash=True)
    finally:
        conn.close()

def update_api_key_last_used(key_id: str, ip: Optional[str] = None) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            if ip:
                conn.execute("UPDATE api_keys SET last_used_at = ?, last_used_ip = ? WHERE id = ?", (now, ip, key_id))
            else:
                conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id))
    except Exception as e:
        logger.warning(f"Failed to update api_key last_used_at: {e}")
    finally:
        conn.close()

def increment_api_key_usage(key_id: str, is_success: bool, ip: Optional[str] = None, duration_ms: float = 0.0) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            if is_success:
                conn.execute(
                    """
                    UPDATE api_keys 
                    SET total_requests = total_requests + 1,
                        successful_requests = successful_requests + 1,
                        last_used_at = ?,
                        last_used_ip = COALESCE(?, last_used_ip)
                    WHERE id = ?
                    """,
                    (now, ip, key_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE api_keys 
                    SET total_requests = total_requests + 1,
                        failed_requests = failed_requests + 1,
                        last_used_at = ?,
                        last_used_ip = COALESCE(?, last_used_ip)
                    WHERE id = ?
                    """,
                    (now, ip, key_id)
                )
    except Exception as e:
        logger.warning(f"Failed to increment api_key usage: {e}")
    finally:
        conn.close()

def list_api_keys() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        return [_normalize_api_key_record(dict(r), include_hash=False) for r in rows]
    finally:
        conn.close()

def list_api_keys_paginated(
    search: str = "",
    status: str = "",
    client: str = "",
    environment: str = "",
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    conn = get_connection()
    try:
        query = "SELECT * FROM api_keys WHERE 1=1"
        params = []

        if search:
            query += " AND (name LIKE ? OR client_name LIKE ? OR key_prefix LIKE ? OR description LIKE ?)"
            s = f"%{search.strip()}%"
            params.extend([s, s, s, s])

        if client and client != "all":
            query += " AND client_name LIKE ?"
            params.append(f"%{client.strip()}%")

        if environment and environment != "all":
            query += " AND environment = ?"
            params.append(environment.strip().lower())

        # Safe sort columns
        allowed_sorts = {
            "name": "name",
            "client_name": "client_name",
            "created_at": "created_at",
            "last_used_at": "last_used_at",
            "total_requests": "total_requests"
        }
        col = allowed_sorts.get(sort_by, "created_at")
        direction = "ASC" if sort_order.lower() == "asc" else "DESC"

        # Fetch all matching candidates to apply real-time status filtering (e.g. expiration)
        rows = conn.execute(f"{query} ORDER BY {col} {direction}", params).fetchall()
        
        normalized = [_normalize_api_key_record(dict(r), include_hash=False) for r in rows]

        if status and status != "all":
            target_status = status.lower()
            normalized = [k for k in normalized if k["status"] == target_status]

        total = len(normalized)
        page_items = normalized[offset : offset + limit]

        return {
            "items": page_items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "total_pages": (total + limit - 1) // limit if total > 0 else 1
        }
    finally:
        conn.close()

def update_api_key(key_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        allowed = [
            "name", "client_name", "description", "environment", "permissions",
            "rate_limit_rpm", "rate_limit_rph", "rate_limit_rpd", "ip_whitelist",
            "status", "is_active", "expires_at"
        ]
        updates = []
        params = []
        for k in allowed:
            if k in fields:
                val = fields[k]
                if k in ("permissions", "ip_whitelist") and not isinstance(val, str):
                    val = json.dumps(val)
                elif k == "is_active":
                    val = 1 if val else 0
                updates.append(f"{k} = ?")
                params.append(val)

        if not updates:
            return get_api_key_record(key_id)

        updates.append("updated_at = ?")
        params.append(now)
        params.append(key_id)

        with conn:
            conn.execute(f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?", params)
        return get_api_key_record(key_id)
    finally:
        conn.close()

def toggle_api_key(key_id: str, is_active: bool) -> bool:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_status = "active" if is_active else "disabled"
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE api_keys SET is_active = ?, status = ?, updated_at = ? WHERE id = ?",
                (1 if is_active else 0, new_status, now, key_id)
            )
            return cur.rowcount > 0
    finally:
        conn.close()

def revoke_api_key(key_id: str) -> bool:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE api_keys SET is_active = 0, status = 'revoked', updated_at = ? WHERE id = ?",
                (now, key_id)
            )
            return cur.rowcount > 0
    finally:
        conn.close()

def get_api_key_stats() -> Dict[str, Any]:
    """Computes dynamic statistics across all API keys and request volume."""
    conn = get_connection()
    try:
        all_keys = list_api_keys()
        total_keys = len(all_keys)
        active_keys = sum(1 for k in all_keys if k["status"] == "active")
        revoked_keys = sum(1 for k in all_keys if k["status"] == "revoked")
        expired_keys = sum(1 for k in all_keys if k["status"] == "expired")
        disabled_keys = sum(1 for k in all_keys if k["status"] == "disabled")

        today_date = time.strftime("%Y-%m-%d")

        # Requests from logs
        total_logs = conn.execute("SELECT COUNT(*) as cnt FROM api_request_logs").fetchone()["cnt"]
        today_logs = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE created_at LIKE ?",
            (f"{today_date}%",)
        ).fetchone()["cnt"]

        succ_logs = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE status_code >= 200 AND status_code < 400"
        ).fetchone()["cnt"]

        failed_logs = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_request_logs WHERE status_code >= 400"
        ).fetchone()["cnt"]

        avg_latency_row = conn.execute("SELECT AVG(processing_time_ms) as avg_ms FROM api_request_logs").fetchone()
        avg_latency = round(avg_latency_row["avg_ms"] or 0.0, 1)

        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "revoked_keys": revoked_keys,
            "expired_keys": expired_keys,
            "disabled_keys": disabled_keys,
            "total_requests": total_logs,
            "successful_requests": succ_logs,
            "failed_requests": failed_logs,
            "requests_today": today_logs,
            "avg_latency_ms": avg_latency,
            "success_rate": round((succ_logs / total_logs * 100), 1) if total_logs > 0 else 100.0
        }
    finally:
        conn.close()

def get_api_key_analytics() -> Dict[str, Any]:
    """Returns analytics for usage by endpoint, key, and daily volume."""
    conn = get_connection()
    try:
        # By endpoint
        ep_rows = conn.execute("""
            SELECT endpoint, COUNT(*) as count, AVG(processing_time_ms) as avg_latency
            FROM api_request_logs
            GROUP BY endpoint
            ORDER BY count DESC LIMIT 10
        """).fetchall()
        by_endpoint = [
            {"endpoint": r["endpoint"], "count": r["count"], "avg_latency_ms": round(r["avg_latency"] or 0.0, 1)}
            for r in ep_rows
        ]

        # By API key
        key_rows = conn.execute("""
            SELECT l.api_key_id, COALESCE(k.name, 'Default / Dev Key') as key_name, COUNT(*) as count
            FROM api_request_logs l
            LEFT JOIN api_keys k ON l.api_key_id = k.id
            GROUP BY l.api_key_id
            ORDER BY count DESC LIMIT 10
        """).fetchall()
        by_key = [
            {"api_key_id": r["api_key_id"], "key_name": r["key_name"], "count": r["count"]}
            for r in key_rows
        ]

        # Volume over past 7 days
        daily_rows = conn.execute("""
            SELECT SUBSTR(created_at, 1, 10) as day, 
                   COUNT(*) as total,
                   SUM(CASE WHEN status_code >= 200 AND status_code < 400 THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as failed
            FROM api_request_logs
            GROUP BY day
            ORDER BY day DESC LIMIT 7
        """).fetchall()
        daily = [
            {"day": r["day"], "total": r["total"], "successful": r["successful"] or 0, "failed": r["failed"] or 0}
            for r in reversed(daily_rows)
        ]

        return {
            "by_endpoint": by_endpoint,
            "by_key": by_key,
            "daily_volume": daily
        }
    finally:
        conn.close()

def list_api_request_logs_paginated(
    api_key_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    status_code: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    conn = get_connection()
    try:
        query = """
            SELECT l.*, COALESCE(k.name, 'Local / Dev Key') as key_name, COALESCE(k.client_name, '') as client_app
            FROM api_request_logs l
            LEFT JOIN api_keys k ON l.api_key_id = k.id
            WHERE 1=1
        """
        params = []

        if api_key_id:
            query += " AND l.api_key_id = ?"
            params.append(api_key_id)

        if endpoint:
            query += " AND l.endpoint LIKE ?"
            params.append(f"%{endpoint.strip()}%")

        if method:
            query += " AND l.method = ?"
            params.append(method.strip().upper())

        if status_code:
            query += " AND l.status_code = ?"
            params.append(status_code)

        if search:
            query += " AND (l.endpoint LIKE ? OR l.client_ip LIKE ? OR k.name LIKE ? OR k.client_name LIKE ?)"
            s = f"%{search.strip()}%"
            params.extend([s, s, s, s])

        count_query = f"SELECT COUNT(*) as cnt FROM ({query})"
        total = conn.execute(count_query, params).fetchone()["cnt"]

        query += f" ORDER BY l.created_at DESC LIMIT {limit} OFFSET {offset}"
        rows = conn.execute(query, params).fetchall()

        items = []
        for r in rows:
            d = dict(r)
            d["duration_ms"] = d.get("processing_time_ms") or 0.0
            items.append(d)

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "total_pages": (total + limit - 1) // limit if total > 0 else 1
        }
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
    timeout_sec: int = 10,
    webhook_type: str = "outgoing",
    description: Optional[str] = None
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
                    is_active, webhook_type, description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    sub_id, integration_id, name, target_url, events_str, secret_key,
                    headers_str, retry_count, retry_interval_sec, timeout_sec,
                    webhook_type, description, now, now
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
        d["webhook_type"] = d.get("webhook_type") or "outgoing"
        d["description"] = d.get("description") or ""
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
            d["webhook_type"] = d.get("webhook_type") or "outgoing"
            d["description"] = d.get("description") or ""
            d["secret_key_preview"] = d["secret_key"][:6] + "..." if d.get("secret_key") else ""
            items.append(d)
        return items
    finally:
        conn.close()

EVENT_ALIASES: Dict[str, List[str]] = {
    "stt.transcription.completed": ["stt.completed"],
    "stt.completed": ["stt.transcription.completed"],
    "stt.transcription.failed": ["stt.failed"],
    "stt.failed": ["stt.transcription.failed"],
    "stt.transcription.started": ["stt.started"],
    "stt.started": ["stt.transcription.started"],
    "tts.generation.completed": ["tts.completed"],
    "tts.completed": ["tts.generation.completed"],
    "tts.generation.started": ["tts.started"],
    "tts.started": ["tts.generation.started"],
    "tts.generation.failed": ["tts.failed"],
    "tts.failed": ["tts.generation.failed"],
    "tts.audio.generated": ["audio.generated"],
    "audio.generated": ["tts.audio.generated"],
    "model.status.changed": ["model.status_changed"],
    "model.status_changed": ["model.status.changed"],
    "webhook.test": ["test.ping"],
    "test.ping": ["webhook.test"]
}

def get_subscriptions_for_event(event_name: str) -> List[Dict[str, Any]]:
    """Retrieves all active subscriptions that subscribed to event_name, aliases, or wildcard '*'."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM webhook_subscriptions WHERE is_active = 1").fetchall()
        matching = []
        target_events = {event_name} | set(EVENT_ALIASES.get(event_name, []))
        for r in rows:
            d = dict(r)
            evts = json.loads(d.get("events") or "[]")
            if "*" in evts or any(e in target_events for e in evts):
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
            "retry_count", "retry_interval_sec", "timeout_sec", "is_active",
            "webhook_type", "description"
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
    error_message: Optional[str] = None,
    request_headers: Optional[Dict[str, str]] = None
) -> str:
    del_id = f"del_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload_str = json.dumps(payload, ensure_ascii=False)
    headers_str = json.dumps(request_headers or {})

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO webhook_deliveries (
                    id, subscription_id, event_name, target_url, payload,
                    http_status, response_body, duration_ms, attempt_count,
                    status, error_message, request_headers, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    del_id, subscription_id, event_name, target_url, payload_str,
                    http_status, response_body[:2000] if response_body else None,
                    round(duration_ms, 2), attempt_count, status, error_message,
                    headers_str, now
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
            query += " AND (event_name = ? OR event_name LIKE ?)"
            params.extend([event_name, f"{event_name}%"])
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
            try:
                d["request_headers"] = json.loads(d.get("request_headers") or "{}")
            except Exception:
                d["request_headers"] = {}
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
        try:
            d["request_headers"] = json.loads(d.get("request_headers") or "{}")
        except Exception:
            d["request_headers"] = {}
        return d
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# CallZenza Integration Specific Helpers
# -----------------------------------------------------------------------------
def get_callzenza_integration() -> Optional[Dict[str, Any]]:
    """Retrieves CallZenza AI Receptionist integration record if present."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM integrations WHERE platform_type = 'callzenza' OR name LIKE '%CallZenza%' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["custom_headers"] = json.loads(d.get("custom_headers") or "{}")
        return d
    finally:
        conn.close()

def get_callzenza_stats() -> Dict[str, Any]:
    """Retrieves live metrics specific to CallZenza webhook delivery."""
    conn = get_connection()
    try:
        cz = get_callzenza_integration()
        if not cz:
            return {
                "connected": False,
                "status": "Disconnected",
                "webhook_url": None,
                "total_deliveries": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "last_event_at": None,
                "last_success_at": None,
                "selected_events": []
            }

        url = cz.get("webhook_url") or cz.get("base_url") or ""
        row = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM webhook_deliveries
            WHERE target_url LIKE ? OR event_name LIKE 'callzenza%'
        """, (f"%{url[:20]}%",) if url else ("%callzenza%",)).fetchone()

        last_deliv = conn.execute("""
            SELECT created_at, event_name, status FROM webhook_deliveries 
            WHERE target_url LIKE ? OR event_name LIKE 'callzenza%'
            ORDER BY created_at DESC LIMIT 1
        """, (f"%{url[:20]}%",) if url else ("%callzenza%",)).fetchone()

        last_success = conn.execute("""
            SELECT created_at FROM webhook_deliveries 
            WHERE (target_url LIKE ? OR event_name LIKE 'callzenza%') AND status = 'success'
            ORDER BY created_at DESC LIMIT 1
        """, (f"%{url[:20]}%",) if url else ("%callzenza%",)).fetchone()

        linked_sub = None
        if url:
            sub_row = conn.execute("SELECT * FROM webhook_subscriptions WHERE target_url = ? LIMIT 1", (url,)).fetchone()
            if sub_row:
                linked_sub = dict(sub_row)
                linked_sub["events"] = json.loads(linked_sub.get("events") or "[]")

        total = row["total"] or 0 if row else 0
        succ = row["successful"] or 0 if row else 0
        fail = row["failed"] or 0 if row else 0

        return {
            "connected": cz.get("is_active", 1) == 1,
            "status": "Connected" if cz.get("is_active", 1) == 1 else "Disconnected",
            "name": cz.get("name", "CallZenza AI Receptionist"),
            "integration_id": cz["id"],
            "webhook_url": url,
            "auth_method": cz.get("auth_method", "hmac_signature"),
            "secret_preview": (cz.get("secret_key_encrypted") or "whsec_callzenza")[:8] + "...",
            "selected_events": linked_sub.get("events", ["*"]) if linked_sub else ["*"],
            "total_deliveries": total,
            "successful_deliveries": succ,
            "failed_deliveries": fail,
            "success_rate": round((succ / total * 100), 1) if total > 0 else 100.0,
            "last_event_at": last_deliv["created_at"] if last_deliv else None,
            "last_event_name": last_deliv["event_name"] if last_deliv else None,
            "last_success_at": last_success["created_at"] if last_success else None
        }
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
    client_ip: Optional[str] = None,
    client_name: Optional[str] = None,
    error_message: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    req_id = f"req_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO api_request_logs (
                    id, api_key_id, client_name, endpoint, method, status_code,
                    processing_time_ms, client_ip, error_message, user_agent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req_id, api_key_id, client_name or "", endpoint, method, status_code,
                    round(processing_time_ms, 2), client_ip or "", error_message or "",
                    user_agent or "", now
                )
            )
        if api_key_id:
            increment_api_key_usage(api_key_id, status_code < 400, client_ip, processing_time_ms)
        return req_id
    except Exception as e:
        logger.warning(f"Failed to log API request: {e}")
        return req_id
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
        total_webhooks = conn.execute("SELECT COUNT(*) as cnt FROM webhook_subscriptions").fetchone()["cnt"]
        active_integrations = conn.execute("SELECT COUNT(*) as cnt FROM integrations WHERE is_active = 1").fetchone()["cnt"]
        active_webhooks = conn.execute("SELECT COUNT(*) as cnt FROM webhook_subscriptions WHERE is_active = 1").fetchone()["cnt"]

        # 5. Webhook delivery statistics (Today and All-Time)
        wh_stats = conn.execute(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM webhook_deliveries
            WHERE created_at LIKE ?
            """,
            (f"{today_date}%",)
        ).fetchone()

        wh_all = conn.execute(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM webhook_deliveries
            """
        ).fetchone()

        total_deliv = wh_all["total"] or 0
        succ_deliv = wh_all["successful"] or 0
        fail_deliv = wh_all["failed"] or 0
        pend_deliv = wh_all["pending"] or 0
        overall_success_rate = round((succ_deliv / total_deliv * 100), 1) if total_deliv > 0 else 100.0

        # 6. Real connectivity tracking for status panel
        last_conn = conn.execute(
            "SELECT created_at, endpoint FROM api_request_logs WHERE status_code < 400 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        last_success_deliv = conn.execute(
            "SELECT created_at, target_url, event_name FROM webhook_deliveries WHERE status = 'success' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        last_failed_deliv = conn.execute(
            "SELECT created_at, target_url, event_name, error_message FROM webhook_deliveries WHERE status = 'failed' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        return {
            "api_requests_today": total_today,
            "stt_requests": stt_today,
            "tts_requests": tts_today,
            "successful_requests": success_today,
            "failed_requests": failed_today,
            "avg_processing_time_ms": avg_processing_time,
            "active_integrations": active_integrations,
            "total_webhooks": total_webhooks,
            "active_webhooks": active_webhooks,
            "active_webhook_subscriptions": active_webhooks,
            "total_deliveries": total_deliv,
            "successful_deliveries": succ_deliv,
            "failed_deliveries": fail_deliv,
            "pending_deliveries": pend_deliv,
            "success_rate": overall_success_rate,
            "webhook_deliveries_today": wh_stats["total"] if wh_stats else 0,
            "webhook_successful_today": wh_stats["successful"] if wh_stats and wh_stats["successful"] else 0,
            "webhook_failed_today": wh_stats["failed"] if wh_stats and wh_stats["failed"] else 0,
            "last_successful_connection": dict(last_conn) if last_conn else None,
            "last_successful_delivery": dict(last_success_deliv) if last_success_deliv else None,
            "last_delivery_failure": dict(last_failed_deliv) if last_failed_deliv else None,
        }
    finally:
        conn.close()

# Auto-initialize database schema on module load
init_db()
