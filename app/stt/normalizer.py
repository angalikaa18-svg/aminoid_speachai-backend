import re

class TextNormalizer:
    """Cleans and standardizes transcribed text output."""

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        # Remove repeated whitespace
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned
