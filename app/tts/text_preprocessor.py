import re
from typing import List

class TextPreprocessor:
    """Cleans, normalizes numbers and segments input text for speech synthesis."""

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove unwanted control characters
        cleaned = re.sub(r"[\r\t]+", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def expand_numbers(text: str) -> str:
        """Simple number expansion rules for common patterns."""
        # Normalize dollar signs
        text = re.sub(r"\$(\d+)", r"\1 dollars", text)
        # Normalize percentages
        text = re.sub(r"(\d+)%", r"\1 percent", text)
        return text

    @classmethod
    def prepare(cls, text: str) -> str:
        cleaned = cls.clean_text(text)
        return cls.expand_numbers(cleaned)
