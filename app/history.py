import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

HISTORY_FILE = Path(__file__).resolve().parent.parent / "storage" / "history.json"

class HistoryManager:
    """Stores local STT and TTS task history on the local filesystem."""

    @classmethod
    def _read_data(cls) -> Dict[str, List[Dict[str, Any]]]:
        if not HISTORY_FILE.exists():
            return {"stt": [], "tts": []}
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"stt": [], "tts": []}

    @classmethod
    def _write_data(cls, data: Dict[str, List[Dict[str, Any]]]) -> None:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def add_stt_entry(cls, entry: Dict[str, Any]) -> None:
        data = cls._read_data()
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["stt"].insert(0, entry)
        data["stt"] = data["stt"][:50] # Keep latest 50
        cls._write_data(data)

    @classmethod
    def add_tts_entry(cls, entry: Dict[str, Any]) -> None:
        data = cls._read_data()
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["tts"].insert(0, entry)
        data["tts"] = data["tts"][:50] # Keep latest 50
        cls._write_data(data)

    @classmethod
    def get_stt_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("stt", [])

    @classmethod
    def get_tts_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("tts", [])
