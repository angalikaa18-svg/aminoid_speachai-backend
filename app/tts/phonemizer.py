class Phonemizer:
    """Pause, punctuation, and rhythm coordinator for local TTS."""

    @staticmethod
    def insert_pause_markers(text: str) -> str:
        # Standardize pause commas and periods for natural SAPI cadence
        text = text.replace(";", ",")
        text = text.replace("—", ", ")
        text = text.replace("...", ".")
        return text
