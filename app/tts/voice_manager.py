from typing import List, Dict, Any, Optional
import pythoncom
import win32com.client
import pyttsx3
from .schemas import TTSVoice

class VoiceManager:
    """Discovers and manages local installed text-to-speech voices."""
    _cached_voices: Optional[List[TTSVoice]] = None

    @classmethod
    def get_available_voices(cls, force_refresh: bool = False) -> List[TTSVoice]:
        if cls._cached_voices is not None and not force_refresh:
            return cls._cached_voices

        voices_list = []
        seen_ids = set()

        # Thread-safe COM initialization for discovering Windows SAPI and OneCore voices
        pythoncom.CoInitialize()
        try:
            # Query OneCore voices & Classic SAPI voices
            categories = [
                (r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", "OneCore"),
                (r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices", "SAPI5"),
            ]

            for cat_path, engine_name in categories:
                try:
                    category = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
                    category.SetId(cat_path, False)
                    tokens = category.EnumerateTokens()
                    for i in range(tokens.Count):
                        t = tokens.Item(i)
                        token_id = str(t.Id)
                        if token_id in seen_ids:
                            continue
                        seen_ids.add(token_id)

                        name = str(t.GetDescription())
                        
                        # Dynamic gender detection from engine metadata (never assumed)
                        gender_val = "unknown"
                        try:
                            g_attr = t.GetAttribute("Gender")
                            if g_attr:
                                gl = str(g_attr).strip().lower()
                                if gl in ["male", "female"]:
                                    gender_val = gl
                        except Exception:
                            pass

                        # Language detection from token attributes / description
                        lang_val = "en"
                        try:
                            if "United Kingdom" in name or "Great Britain" in name:
                                lang_val = "en-GB"
                            elif "United States" in name or "US" in name:
                                lang_val = "en-US"
                            elif "India" in name:
                                lang_val = "en-IN"
                            else:
                                l_attr = t.GetAttribute("Language")
                                if l_attr:
                                    lang_val = str(l_attr)
                        except Exception:
                            pass

                        # Age detection from token attributes
                        age_val = None
                        try:
                            age_attr = t.GetAttribute("Age")
                            if age_attr:
                                age_val = str(age_attr)
                        except Exception:
                            pass

                        voices_list.append(TTSVoice(
                            id=token_id,
                            name=name,
                            gender=gender_val,
                            language=lang_val,
                            age=age_val,
                            engine=engine_name
                        ))
                    del tokens
                    del category
                except Exception:
                    pass

        except Exception:
            pass
        finally:
            pythoncom.CoUninitialize()

        # Fallback to pyttsx3 if COM enumeration was empty
        if not voices_list:
            try:
                engine = pyttsx3.init()
                installed = engine.getProperty("voices") or []
                for v in installed:
                    # Check gender from attribute if engine provided it
                    g_val = getattr(v, "gender", None)
                    gender = "unknown"
                    if g_val:
                        gl = str(g_val).lower()
                        if gl in ["male", "female"]:
                            gender = gl

                    voices_list.append(TTSVoice(
                        id=str(v.id),
                        name=str(v.name),
                        gender=gender,
                        language="en",
                        age=getattr(v, "age", None),
                        engine="SAPI5"
                    ))
                engine.stop()
            except Exception:
                voices_list.append(TTSVoice(
                    id="default",
                    name="System Default Voice",
                    gender="unknown",
                    language="en",
                    engine="SAPI5"
                ))

        cls._cached_voices = voices_list
        return voices_list

    @classmethod
    def find_voice(cls, voice_id: Optional[str]) -> Optional[str]:
        if not voice_id:
            return None
        voices = cls.get_available_voices()
        for v in voices:
            if v.id == voice_id or v.name.lower() == voice_id.lower():
                return v.id
        return None
