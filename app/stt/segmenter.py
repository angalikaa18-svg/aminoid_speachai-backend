from typing import List, Dict, Any
from .schemas import Segment, WordTimestamp

class Segmenter:
    """Processes raw Faster-Whisper segments and word timestamps."""

    @staticmethod
    def process_segments(raw_segments) -> List[Dict[str, Any]]:
        processed = []
        for idx, seg in enumerate(raw_segments):
            words_data = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3)
                    })

            processed.append({
                "id": idx,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "confidence": round(seg.avg_logprob, 3) if hasattr(seg, "avg_logprob") else 1.0,
                "speaker": None,
                "words": words_data
            })
        return processed
