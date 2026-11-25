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
    name: str = Field(..., description="Friendly name for this key, e.g. 'CallZenza AI Receptionist'")
    client_name: Optional[str] = Field("CallZenza AI Receptionist", description="Client or application name")
    client_id: Optional[str] = Field(None, description="Client identifier")
    description: Optional[str] = Field(None, description="Description or integration notes")
    environment: Optional[str] = Field("production", description="Environment: development, testing, or production")
    permissions: Optional[List[str]] = Field(
        default=["stt.transcribe", "tts.generate", "tts.voices.read", "audio.read"],
        description="Allowed permission scopes"
    )
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650, description="Expiration in days (null for perpetual)")
    custom_expiration_date: Optional[str] = Field(None, description="ISO format custom expiration date YYYY-MM-DD")
    rate_limit_rpm: Optional[int] = Field(60, ge=1, le=10000, description="Max requests per minute")
    rate_limit_rph: Optional[int] = Field(1000, ge=1, le=100000, description="Max requests per hour")
    rate_limit_rpd: Optional[int] = Field(10000, ge=1, le=1000000, description="Max requests per day")
    ip_whitelist: Optional[List[str]] = Field(default_factory=list, description="Allowed IP addresses (empty for unrestricted)")

class ApiKeyUpdateRequest(BaseModel):
    name: Optional[str] = None
    client_name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None
    rate_limit_rpm: Optional[int] = None
    rate_limit_rph: Optional[int] = None
    rate_limit_rpd: Optional[int] = None
    ip_whitelist: Optional[List[str]] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None

class ApiKeyTestRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="Raw or bearer API key to test")
    apiKey: Optional[str] = Field(None, description="CamelCase alias for API key")
    key: Optional[str] = Field(None, description="Short alias for API key")
    key_id: Optional[str] = Field(None, description="Key ID alias")
    target_key: Optional[str] = Field(None, description="Target key alias")

    model_config = {
        "extra": "ignore"
    }

    @property
    def effective_key(self) -> str:
        return (self.api_key or self.apiKey or self.key or self.key_id or self.target_key or "").strip()

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
    timeout_sec: Optional[float] = Field(5.0, ge=0.5, le=60.0)

class TestConnectionRequest(BaseModel):
    base_url: Optional[str] = None
    url: Optional[str] = None
    target_url: Optional[str] = None
    auth_method: Optional[str] = Field("none")
    api_key: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)

    @property
    def effective_url(self) -> str:
        return (self.base_url or self.url or self.target_url or "").strip()

class ExternalTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize into speech")
    voice_id: Optional[str] = Field(None, description="Voice ID from /api/v1/integrations/voices")
    language: Optional[str] = Field("en", description="Target language")
    format: Optional[str] = Field("wav", description="Audio container format")
    rate: Optional[int] = Field(175, ge=50, le=400, description="Speaking speed rate")
    volume: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Audio volume")
    pitch: Optional[int] = Field(50, ge=0, le=100, description="Pitch modulation")
    response_mode: Optional[str] = Field("json", description="'json' for URL & metadata, 'audio' for raw audio/wav stream")

class DiagnosticsRunRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="Optional raw API key to test authentication")
    target_platform_url: Optional[str] = Field(None, description="Optional platform URL to ping")
    target_webhook_url: Optional[str] = Field(None, description="Optional webhook URL to test delivery")
    webhook_secret: Optional[str] = Field(None, description="Optional HMAC secret for webhook test")

class CallZenzaConnectRequest(BaseModel):
    destination_url: str = Field(..., description="CallZenza webhook incoming endpoint URL")
    secret_key: Optional[str] = Field(None, description="HMAC secret key for CallZenza signature verification")
    auth_token: Optional[str] = Field(None, description="Optional Bearer token or API key for CallZenza header")
    events: Optional[List[str]] = Field(None, description="Events to subscribe to (defaults to all speech events)")
    base_url: Optional[str] = Field(None, description="CallZenza server base URL")
    description: Optional[str] = Field("CallZenza AI Receptionist telephony integration", description="Notes")

class CallZenzaTestRequest(BaseModel):
    event_name: Optional[str] = Field("callzenza.session.started", description="Event name to dispatch")
    sample_payload: Optional[Dict[str, Any]] = Field(None, description="Custom payload for test")

class CallZenzaSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Unique session/call ID")
    caller_number: Optional[str] = Field(None, description="Caller phone number")
    agent_name: Optional[str] = Field("CallZenza AI Receptionist", description="Name of receptionist agent")
    initial_prompt: Optional[str] = Field("Hello, thank you for calling. How may I assist you today?", description="Greeting prompt")

