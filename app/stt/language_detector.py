from typing import Dict

# ISO 639-1 language code to readable name mapping
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian",
    "vi": "Vietnamese", "sv": "Swedish", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "fi": "Finnish", "he": "Hebrew"
}

class LanguageDetector:
    @staticmethod
    def get_language_name(code: str) -> str:
        return LANGUAGE_NAMES.get(code.lower(), code.upper())
