"""Standalone Offline Platform Test Runner

Executes comprehensive tests without invoking vendor SDK conftest / Docker WireMock.
"""

import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.offline import test_offline_platform as t

def run_all_tests():
    test_functions = [
        ("Health Endpoint", t.test_health_endpoint),
        ("HTML Dashboard", t.test_index_dashboard),
        ("Offline STT Audio Transcription", t.test_transcribe_endpoint_valid_audio),
        ("Empty Audio Validation", t.test_transcribe_empty_audio),
        ("Unsupported Format Validation", t.test_transcribe_unsupported_format),
        ("SSRF URL Blocking", t.test_transcribe_url_blocked_for_privacy),
        ("Offline TTS Voices Discovery", t.test_tts_voices_list),
        ("Offline TTS Speech Generation", t.test_tts_generate_endpoint),
        ("Local Models Status", t.test_models_status),
        ("Zero Deepgram / Cloud Privacy Check", t.test_zero_deepgram_in_app)
    ]

    print("=" * 65)
    print("  RUNNING OFFLINE PLATFORM TEST SUITE")
    print("=" * 65)

    passed = 0
    failed = 0

    for name, func in test_functions:
        print(f"[*] Running: {name} ...", end=" ", flush=True)
        try:
            func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print("✗ FAILED")
            traceback.print_exc()
            failed += 1

    print("=" * 65)
    print(f"  RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 65)

    if failed > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
