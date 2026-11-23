"""
Pydantic Request & Response Models for AMINOID Speech Integrations.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class PlatformCreateRequest(BaseModel):
    name: str = Field(..., description="Name of external platform or application")
    platform_type: str = Field("custom_app", description="Platform category")
    description: Optional[str] = Field(None, description="Optional notes or description")
    base_url: Optional[str] = Field(None, description="Platform base URL for outgoing requests or API ping")
    webhook_url: Optional[str] = Field(None, description="Platform destination URL for webhook events")
    auth_method: Optional[str] = Field("none", description="Auth method: none, api_key, bearer_token, custom_header, hmac_signature")
    api_key: Optional[str] = Field(None, description="Outbound API Key for platform")
    secret_key: Optional[str] = Field(None, description="Outbound Secret Key")
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    timeout_sec: Optional[int] = Field(30, ge=1, le=120)

class PlatformUpdateRequest(BaseModel):
    name: Optional[str] = None
    platform_type: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    webhook_url: Optional[str] = None
    auth_method: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None
    timeout_sec: Optional[int] = None
    is_active: Optional[bool] = None

class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., description="Friendly name for this key, e.g. 'CRM Voice Agent'")
    permissions: Optional[List[str]] = Field(
        default=["stt:transcribe", "tts:generate", "voices:read", "audio:read"],
        description="Allowed permission scopes"
    )
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650, description="Expiration in days (null for perpetual)")

class WebhookSubscriptionCreateRequest(BaseModel):
    name: str = Field(..., description="Name for this subscription")
    target_url: str = Field(..., description="Destination HTTP/HTTPS webhook URL")
    events: List[str] = Field(..., description="List of event names to subscribe to, or ['*'] for all")
    secret_key: Optional[str] = Field(None, description="HMAC-SHA256 signature secret (auto-generated if omitted)")
    integration_id: Optional[str] = Field(None, description="Optional linked integration ID")
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    retry_count: Optional[int] = Field(3, ge=0, le=10)
    retry_interval_sec: Optional[int] = Field(5, ge=1, le=60)
    timeout_sec: Optional[int] = Field(10, ge=1, le=60)

class WebhookSubscriptionUpdateRequest(BaseModel):
    name: Optional[str] = None
    target_url: Optional[str] = None
    events: Optional[List[str]] = None
    secret_key: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None
    retry_count: Optional[int] = None
    retry_interval_sec: Optional[int] = None
    timeout_sec: Optional[int] = None
    is_active: Optional[bool] = None

class TestWebhookRequest(BaseModel):
    target_url: str = Field(..., description="Target destination URL")
    secret_key: Optional[str] = Field(None, description="Secret key for HMAC signature")
    event_name: Optional[str] = Field("test.ping", description="Event name to simulate")
    sample_payload: Optional[Dict[str, Any]] = None
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)

class TestConnectionRequest(BaseModel):
    base_url: str = Field(..., description="URL to ping or test")
    auth_method: Optional[str] = Field("none")
    api_key: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)

class ExternalTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize into speech")
    voice_id: Optional[str] = Field(None, description="Voice ID from /api/v1/integrations/voices")
    language: Optional[str] = Field("en", description="Target language")
    format: Optional[str] = Field("wav", description="Audio container format")
    rate: Optional[int] = Field(175, ge=50, le=400, description="Speaking speed rate")
    volume: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Audio volume")
    pitch: Optional[int] = Field(50, ge=0, le=100, description="Pitch modulation")
    response_mode: Optional[str] = Field("json", description="'json' for URL & metadata, 'audio' for raw audio/wav stream")
