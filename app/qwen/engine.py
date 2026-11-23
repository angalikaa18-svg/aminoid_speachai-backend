"""
Qwen3-TTS Neural Voice Cloning Engine.
Coordinates:
- Model lifecycle management (CPU-compatible loading with torch.float32 / sdpa attention)
- Acoustic preprocessing and validation (24kHz sinc resample, peak normalized, silence trimmed)
- Reference transcript handling & Whisper auto-transcription
- Speaker conditioning extraction (create_voice_clone_prompt with ICL / x-vector)
- Neural synthesis (generate_voice_clone)
- Objective speaker similarity evaluation (MFCC + Spectral Contrast cosine score)
- Phase 9 structured diagnostic telemetry
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch
import librosa

from .audio_validator import validate_audio_file, preprocess_reference_audio
from .similarity import evaluate_speaker_similarity
from app.tts.config import DEFAULT_TTS_CONFIG
from app.history import HistoryManager

logger = logging.getLogger("QwenVoiceEngine")

# Supported models
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
HIGH_CAPACITY_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

class QwenVoiceEngine:
    """Singleton manager for Qwen3-TTS Neural Cloning and Synthesis."""
    _instance: Optional["QwenVoiceEngine"] = None

    def __init__(self):
        self.active_model_id = DEFAULT_MODEL_ID
        self.device = "cpu"
        self.dtype = torch.float32
        self.model = None
        self.is_loading = False
        self.load_error = None
        self.loaded_at = None

    @classmethod
    def get_instance(cls) -> "QwenVoiceEngine":
        if cls._instance is None:
            cls._instance = QwenVoiceEngine()
        return cls._instance

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_id": self.active_model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "is_loaded": self.model is not None,
            "is_loading": self.is_loading,
            "load_error": self.load_error,
            "loaded_at": self.loaded_at,
            "engine": "Qwen3-TTS Neural Engine"
        }

    def load_model(self, model_id: Optional[str] = None) -> bool:
        """
        Loads the Qwen3-TTS model in CPU-safe mode with local_files_only=True.
        """
        target_model = model_id or self.active_model_id
        if self.model is not None and self.active_model_id == target_model:
            return True

        self.is_loading = True
        self.load_error = None

        try:
            # Leave at least 1-2 CPU cores available for Uvicorn event loop and health checks
            available_cpus = os.cpu_count() or 4
            safe_threads = max(1, available_cpus - 1)
            torch.set_num_threads(safe_threads)
            logger.info(f"Loading Qwen3-TTS model '{target_model}' on CPU with float32 precision ({safe_threads} threads, local offline mode)...")
            from qwen_tts import Qwen3TTSModel

            # CPU-safe parameters
            tts_model = Qwen3TTSModel.from_pretrained(
                target_model,
                device_map="cpu",
                dtype=self.dtype,
                attn_implementation="sdpa",
                local_files_only=True
            )

            self.model = tts_model
            self.active_model_id = target_model
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info(f"Qwen3-TTS model '{target_model}' loaded successfully on CPU.")
            return True

        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            return False
        finally:
            self.is_loading = False

    def clone_voice(
        self,
        ref_audio_path: str,
        target_text: str,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        model_id: Optional[str] = None,
        language: str = "English",
        voice_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Neural Voice Cloning:
        1. Validates and preprocesses reference audio (24kHz sinc resampled, peak normalized)
        2. Handles reference transcript (uses provided, or auto-transcribes with local Faster-Whisper,
           or falls back to speaker embedding only if no speech text is detected)
        3. Builds voice-clone conditioning items (speaker embedding + speech codec tokens)
        4. Synthesizes target text strictly using Qwen3-TTS Base model
        5. Evaluates objective acoustic speaker similarity between reference and output
        6. Logs Phase 9 structured diagnostic report
        7. Saves output WAV and registers history
        """
        req_id = f"clone_{uuid.uuid4().hex[:8]}"
        req_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.time()

        # Step 1: Preprocessing & Validation
        ref_proc_dir = Path(DEFAULT_TTS_CONFIG.output_dir) / "ref_proc"
        ref_proc_dir.mkdir(parents=True, exist_ok=True)
        processed_ref_path = str(ref_proc_dir / f"proc_{Path(ref_audio_path).stem}.wav")

        proc_path, proc_meta = preprocess_reference_audio(ref_audio_path, processed_ref_path)
        validation_info = proc_meta["validation"]

        # Step 2: Reference Transcript Handling & Mode Resolution
        transcript_source = "None"
        clean_ref_text = ref_text.strip() if ref_text else ""

        if not x_vector_only_mode:
            if clean_ref_text:
                transcript_source = "User Provided"
            else:
                # Auto-transcribe using real local Faster-Whisper
                try:
                    from app.stt.engine import LocalSTTEngine
                    stt = LocalSTTEngine()
                    trans_res = stt.transcribe(proc_path)
                    detected_text = trans_res.get("text", "").strip()
                    if detected_text:
                        clean_ref_text = detected_text
                        transcript_source = "Faster-Whisper Auto-Transcribed"
                        logger.info(f"[{req_id}] Auto-transcribed reference audio: '{clean_ref_text}'")
                    else:
                        transcript_source = "Auto-Transcribe (No Speech Detected)"
                except Exception as stt_err:
                    logger.warning(f"[{req_id}] Auto-transcription failed: {stt_err}")
                    transcript_source = "Auto-Transcribe Failed"

                # If still no reference text available, switch to x_vector_only_mode
                if not clean_ref_text:
                    logger.info(f"[{req_id}] No reference text available; switching to speaker embedding only mode (x-vector).")
                    x_vector_only_mode = True
        else:
            transcript_source = "None (x-vector only mode selected)"
            clean_ref_text = None

        # Step 3: Model Loading & Prompt Extraction
        model_loaded = (self.model is not None)
        if not model_loaded:
            model_loaded = self.load_model(model_id or self.active_model_id)

        if not model_loaded or self.model is None:
            raise RuntimeError(f"Qwen3-TTS neural engine could not be loaded: {self.load_error or 'Unknown initialization error'}")

        # Step 4: Speaker Conditioning Extraction
        prompt_ref_text = clean_ref_text if not x_vector_only_mode else None
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=proc_path,
            ref_text=prompt_ref_text,
            x_vector_only_mode=x_vector_only_mode
        )

        item = prompt_items[0]
        spk_emb_shape = list(item.ref_spk_embedding.shape) if hasattr(item.ref_spk_embedding, "shape") else [512]
        ref_code_frames = item.ref_code.shape[0] if item.ref_code is not None else 0

        # Step 5: Neural Generation
        # Ensure thread allocation leaves headroom for server health check responsiveness
        available_cpus = os.cpu_count() or 4
        safe_threads = max(1, available_cpus - 1)
        torch.set_num_threads(safe_threads)

        out_dir = Path(DEFAULT_TTS_CONFIG.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"qwen_{req_id}.wav"
        out_filepath = str(out_dir / out_filename)

        num_words = max(1, len(target_text.strip().split()))
        max_tokens = min(320, max(60, int(num_words * 12) + 36))

        # Auto-resolve language to English if Auto or empty and text is Latin
        synth_language = language
        if not synth_language or str(synth_language).strip().lower() in ("auto", "none"):
            is_latin = all(ord(c) < 128 for c in target_text.strip())
            synth_language = "English" if is_latin else "Auto"

        t_gen_start = time.time()
        wavs, synth_sr = self.model.generate_voice_clone(
            text=target_text.strip(),
            language=synth_language,
            voice_clone_prompt=prompt_items,
            non_streaming_mode=True,
            max_new_tokens=max_tokens
        )
        gen_time = round(time.time() - t_gen_start, 2)

        gen_wav = wavs[0]
        sf.write(out_filepath, gen_wav, synth_sr, subtype="PCM_16")
        duration_sec = round(float(len(gen_wav) / synth_sr), 2)
        total_time = round(time.time() - t0, 2)
        rtf = round(gen_time / max(0.1, duration_sec), 2)

        # Step 6: Objective Speaker Similarity Evaluation (Phase 10)
        similarity_eval = evaluate_speaker_similarity(proc_path, out_filepath)

        # Step 7: Log to History
        HistoryManager.add_tts_entry({
            "text": target_text.strip()[:80],
            "filename": out_filename,
            "duration": duration_sec,
            "voice": f"Cloned ({voice_name or 'Reference Speaker'})"
        })

        # Step 8: Phase 9 Structured Diagnostic Logging
        mode_label = "Speaker Embedding Only (x-vector)" if x_vector_only_mode else "In-Context Learning (ICL)"
        diagnostic_banner = f"""
=================== VOICE CLONING DIAGNOSTICS ===================
Request ID:               {req_id}
Timestamp:                {req_timestamp}
Selected Model:           {model_id or self.active_model_id}
Cloning Mode:             {mode_label}
Reference Audio:
  - Original Path:        {ref_audio_path}
  - Processed Path:       {proc_path}
  - Duration:             {proc_meta["duration"]}s (raw: {validation_info.get("duration", "N/A")}s)
  - Resampled:            {proc_meta["sample_rate"]} Hz, {proc_meta["channels"]} channel (mono)
  - RMS Energy:           {validation_info.get("rms_energy", "N/A")} (Peak: {validation_info.get("peak_amplitude", "N/A")})
  - Clipping Detected:    {validation_info.get("is_clipping", False)}
Reference Transcript:
  - Source:               {transcript_source}
  - Characters:           {len(clean_ref_text) if clean_ref_text else 0} (Words: {len(clean_ref_text.split()) if clean_ref_text else 0})
  - Content:              "{clean_ref_text if clean_ref_text else 'N/A'}"
Speaker Conditioning:
  - Speaker Embedding:    Shape {spk_emb_shape} (1024-dim Qwen3 Speaker Encoder - Attentive Statistics Pooling)
  - Speech Code Tokens:   {ref_code_frames} frames (12Hz discrete tokenizer)
  - ICL Conditioning:     {bool(not x_vector_only_mode)}
  - x-vector Conditioning: {bool(x_vector_only_mode)}
Synthesis Parameters:
  - Target Text:          "{target_text.strip()}"
  - Target Words:         {num_words}
  - Max Codec Tokens:     {max_tokens}
  - Language:             {synth_language}
Generation:
  - Execution Device:     {self.device.upper()} ({str(self.dtype)}, SDPA attention)
  - Output Audio Duration: {duration_sec}s
  - Inference Time:       {gen_time}s
  - Total Pipeline Time:  {total_time}s
  - Real-Time Factor:     {rtf}x
Objective Speaker Similarity:
  - Acoustic Cosine Score: {similarity_eval.get("raw_cosine", "N/A")}
  - Calibrated Match:     {similarity_eval.get("similarity_percentage", "N/A")}%
  - Similarity Rating:    {similarity_eval.get("rating", "N/A")}
  - Assessment:           {similarity_eval.get("description", "N/A")}
Output Deliverable:
  - Output File:          {out_filename}
  - Direct URL:           /api/v1/tts/audio/{out_filename}
  - Engine:               Qwen3-TTS Neural Acoustic Pipeline (100% Local)
=================================================================
"""
        print(diagnostic_banner)
        logger.info(diagnostic_banner)

        return {
            "status": "success",
            "request_id": req_id,
            "filename": out_filename,
            "audio_url": f"/api/v1/tts/audio/{out_filename}",
            "full_audio_url": f"http://127.0.0.1:8080/api/v1/tts/audio/{out_filename}",
            "duration": duration_sec,
            "voice_name": voice_name or f"Cloned Speaker ({Path(ref_audio_path).stem})",
            "model_used": model_id or self.active_model_id,
            "mode": "x_vector_only" if x_vector_only_mode else "icl",
            "used_neural_model": True,
            "validation": validation_info,
            "preprocessing": {
                "sample_rate": proc_meta["sample_rate"],
                "duration": proc_meta["duration"],
                "channels": proc_meta["channels"]
            },
            "transcript_info": {
                "source": transcript_source,
                "text": clean_ref_text,
                "length": len(clean_ref_text) if clean_ref_text else 0
            },
            "conditioning": {
                "speaker_embedding_dim": spk_emb_shape[-1] if spk_emb_shape else 512,
                "speech_code_frames": ref_code_frames,
                "mode": mode_label
            },
            "performance": {
                "inference_time_sec": gen_time,
                "total_time_sec": total_time,
                "rtf": rtf,
                "device": self.device
            },
            "similarity": similarity_eval,
            "offline": True,
            "message": "Voice cloning completed with neural acoustic speaker conditioning."
        }
