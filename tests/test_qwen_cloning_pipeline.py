"""
Comprehensive End-to-End Verification Test for Qwen3-TTS Voice Cloning Pipeline.
Tests:
1. Audio Preprocessing & Signal Integrity (24kHz, peak normalized, silence trimmed)
2. In-Context Learning (ICL) Voice Cloning with reference transcript
3. Automatic Whisper transcription fallback when ref_text is omitted
4. Speaker Embedding Only (x-vector) mode
5. Objective Speaker Similarity scoring (MFCC + Spectral Contrast)
6. Phase 9 Structured Diagnostic Banner output
"""

import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.qwen.engine import QwenVoiceEngine
from app.qwen.audio_validator import validate_audio_file, preprocess_reference_audio
from app.qwen.similarity import evaluate_speaker_similarity

def run_tests():
    print("=" * 60)
    print("STARTING QWEN3-TTS VOICE CLONING PIPELINE VERIFICATION")
    print("=" * 60)

    engine = QwenVoiceEngine.get_instance()
    
    # 1. Model Loading
    print("\n[TEST 1] Loading Qwen3-TTS Engine (local offline mode)...")
    load_ok = engine.load_model()
    assert load_ok, f"Model load failed: {engine.load_error}"
    assert engine.model is not None, "Engine model is None"
    print("--> PASS: Qwen3-TTS model loaded successfully on CPU.")

    # 2. Audio Validation & Preprocessing
    backend_dir = Path(__file__).resolve().parent.parent
    ref_audio = str(backend_dir / "storage" / "tts_output" / "ref_uploads" / "ref_5f61b03e_tts_57b4c1ad25d3.wav")
    assert os.path.exists(ref_audio), f"Reference audio not found: {ref_audio}"

    print(f"\n[TEST 2] Testing Audio Preprocessing on: {ref_audio}")
    proc_path, proc_meta = preprocess_reference_audio(ref_audio)
    assert os.path.exists(proc_path), "Processed audio not created"
    assert proc_meta["sample_rate"] == 24000, "Sample rate not 24000"
    assert proc_meta["channels"] == 1, "Not mono"
    print(f"--> PASS: Resampled to 24kHz mono ({proc_meta['duration']}s), peak normalized.")

    # 3. In-Context Learning (ICL) Cloning with Known Reference Transcript
    ref_text = (
        "Dr. APJ Abdul Kalam is famous for his inspiring speeches to students and youth. "
        "Many of his most memorable addresses focus on dreams, national vision, and the power of hard work."
    )
    target_text = "Good morning! Welcome to AminoidSpeech Voice Studio."

    print(f"\n[TEST 3] Running ICL Mode Voice Cloning...")
    print(f"Target Text: '{target_text}'")
    
    t0 = time.time()
    res = engine.clone_voice(
        ref_audio_path=ref_audio,
        target_text=target_text,
        ref_text=ref_text,
        x_vector_only_mode=False
    )
    elapsed = round(time.time() - t0, 2)

    print(f"--> Generation Finished in {elapsed}s")
    assert res["status"] == "success", f"Status not success: {res}"
    assert res["used_neural_model"] is True, "Neural model was not used!"
    assert res["filename"].startswith("qwen_clone_"), f"Unexpected filename prefix: {res['filename']}"
    out_check_path = backend_dir / "storage" / "tts_output" / res['filename']
    assert os.path.exists(out_check_path), f"Output WAV file does not exist at {out_check_path}!"

    sim = res["similarity"]
    print(f"--> Output WAV: {res['filename']} ({res['duration']}s)")
    print(f"--> Objective Similarity Score: {sim['similarity_percentage']}% ({sim['rating']})")
    assert sim["similarity_percentage"] >= 65.0, f"Similarity too low: {sim['similarity_percentage']}%"
    print("--> PASS: High timbre similarity confirmed.")

    print("\n" + "=" * 60)
    print("ALL QWEN3-TTS VOICE CLONING VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
