import json
from pathlib import Path
from typing import Dict, Any

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "storage" / "settings.json"

DEFAULT_SETTINGS = {
    "api_base_url": "http://127.0.0.1:8080",
    "default_stt_language": "auto",
    "default_vad_filter": True,
    "default_tts_rate": 175,
    "default_tts_volume": 1.0,
    "default_tts_pitch": 50,
    "default_voice": "default",
    "offline_strict_mode": True,
    "auto_webhook_dispatch": True,
    "max_history_items": 100,
    "theme": "dark-green"
}

class SettingsManager:
    """Manages application settings stored persistently in storage/settings.json."""

    @classmethod
    def _read_raw(cls) -> Dict[str, Any]:
        if not SETTINGS_FILE.exists():
            return {}
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        raw = cls._read_raw()
        merged = dict(DEFAULT_SETTINGS)
        merged.update(raw)
        return merged

    @classmethod
    def save_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        current = cls.get_settings()
        current.update(settings)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        return current

    @classmethod
    def reset_settings(cls) -> Dict[str, Any]:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return dict(DEFAULT_SETTINGS)
