from typing import List, Dict, Any

class TimestampProcessor:
    """Formats seconds into standard SubRip (SRT) and VTT timestamps."""

    @staticmethod
    def format_srt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis = 999
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @classmethod
    def generate_srt(cls, segments: List[Dict[str, Any]]) -> str:
        """Construct valid SRT subtitle block from list of segments."""
        if not segments:
            return ""
        lines = []
        for idx, seg in enumerate(segments, 1):
            start_str = cls.format_srt_timestamp(seg.get("start", 0.0))
            end_str = cls.format_srt_timestamp(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            if not text:
                continue
            lines.append(str(idx))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines).strip()
