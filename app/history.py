import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

HISTORY_FILE = Path(__file__).resolve().parent.parent / "storage" / "history.json"

class HistoryManager:
    """
    Stores local STT, TTS, Voice Cloning, and Voice Design task history
    persistently in storage/history.json with unique IDs and CRUD operations.
    """

    @classmethod
    def _read_data(cls) -> Dict[str, List[Dict[str, Any]]]:
        if not HISTORY_FILE.exists():
            return {"stt": [], "tts": [], "clone": [], "design": []}
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                for k in ["stt", "tts", "clone", "design"]:
                    if k not in data or not isinstance(data[k], list):
                        data[k] = []
                return data
        except Exception:
            return {"stt": [], "tts": [], "clone": [], "design": []}

    @classmethod
    def _write_data(cls, data: Dict[str, List[Dict[str, Any]]]) -> None:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def add_stt_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = cls._read_data()
        entry_record = dict(entry)
        if "id" not in entry_record:
            entry_record["id"] = f"stt_{uuid.uuid4().hex[:10]}"
        entry_record["timestamp"] = entry_record.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
        data["stt"].insert(0, entry_record)
        data["stt"] = data["stt"][:100]  # Keep latest 100
        cls._write_data(data)
        return entry_record

    @classmethod
    def add_tts_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = cls._read_data()
        entry_record = dict(entry)
        if "id" not in entry_record:
            entry_record["id"] = f"tts_{uuid.uuid4().hex[:10]}"
        entry_record["timestamp"] = entry_record.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
        data["tts"].insert(0, entry_record)
        data["tts"] = data["tts"][:100]  # Keep latest 100
        cls._write_data(data)
        return entry_record

    @classmethod
    def add_clone_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = cls._read_data()
        entry_record = dict(entry)
        if "id" not in entry_record:
            entry_record["id"] = f"clone_{uuid.uuid4().hex[:10]}"
        entry_record["timestamp"] = entry_record.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
        data["clone"].insert(0, entry_record)
        data["clone"] = data["clone"][:100]
        cls._write_data(data)
        return entry_record

    @classmethod
    def add_design_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = cls._read_data()
        entry_record = dict(entry)
        if "id" not in entry_record:
            entry_record["id"] = f"dsgn_{uuid.uuid4().hex[:10]}"
        entry_record["timestamp"] = entry_record.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
        data["design"].insert(0, entry_record)
        data["design"] = data["design"][:100]
        cls._write_data(data)
        return entry_record

    @classmethod
    def get_stt_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("stt", [])

    @classmethod
    def get_tts_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("tts", [])

    @classmethod
    def get_clone_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("clone", [])

    @classmethod
    def get_design_history(cls) -> List[Dict[str, Any]]:
        return cls._read_data().get("design", [])

    @classmethod
    def get_history(
        cls,
        category: str = "all",
        q: str = "",
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        data = cls._read_data()
        if category in data:
            items = data[category]
        elif category == "all":
            # Combine all categories with a category tag
            items = []
            for cat, cat_items in data.items():
                for item in cat_items:
                    item_copy = dict(item)
                    item_copy["category"] = cat
                    items.append(item_copy)
            # Sort by timestamp desc
            items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        else:
            items = []

        # Apply search query
        if q:
            term = q.lower()
            filtered = []
            for item in items:
                searchable = " ".join(str(v) for v in item.values()).lower()
                if term in searchable:
                    filtered.append(item)
            items = filtered

        total = len(items)
        paginated = items[offset : offset + limit]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": paginated
        }

    @classmethod
    def delete_entry(cls, category: str, entry_id: str) -> bool:
        data = cls._read_data()
        deleted = False
        categories = [category] if category in data else ["stt", "tts", "clone", "design"]
        for cat in categories:
            original_len = len(data[cat])
            data[cat] = [item for item in data[cat] if item.get("id") != entry_id]
            if len(data[cat]) < original_len:
                deleted = True
        if deleted:
            cls._write_data(data)
        return deleted

    @classmethod
    def clear_category(cls, category: str) -> bool:
        data = cls._read_data()
        if category == "all":
            data = {"stt": [], "tts": [], "clone": [], "design": []}
            cls._write_data(data)
            return True
        elif category in data:
            data[category] = []
            cls._write_data(data)
            return True
        return False

    @classmethod
    def get_counts(cls) -> Dict[str, int]:
        data = cls._read_data()
        return {
            "stt": len(data.get("stt", [])),
            "tts": len(data.get("tts", [])),
            "clone": len(data.get("clone", [])),
            "design": len(data.get("design", [])),
            "total": sum(len(v) for v in data.values())
        }
