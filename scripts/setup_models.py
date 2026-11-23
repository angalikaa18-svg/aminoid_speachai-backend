"""Explicit Model Setup Script

Downloads and verifies approved local offline STT model weights.
This is an explicit offline preparation step. Once run, the platform operates
completely without internet access.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def setup_stt_model(model_name: str = "tiny"):
    target_dir = ROOT_DIR / "models" / "stt" / f"faster-whisper-{model_name}"
    print(f"[*] Setting up local STT model '{model_name}' at: {target_dir}")
    
    if (target_dir / "model.bin").exists():
        print(f"[+] Model '{model_name}' is already installed locally!")
        return str(target_dir)

    try:
        from faster_whisper import download_model
        print(f"[*] Downloading '{model_name}' model weights to local models folder...")
        path = download_model(model_name, output_dir=str(target_dir))
        print(f"[+] Successfully installed offline STT model at: {path}")
        return path
    except Exception as e:
        print(f"[-] Failed to download model: {e}")
        sys.exit(1)

def verify_models():
    stt_dir = ROOT_DIR / "models" / "stt" / "faster-whisper-tiny"
    if (stt_dir / "model.bin").exists():
        print(f"[✓] STT Model verified: {stt_dir} (model.bin present)")
    else:
        print(f"[✗] STT Model missing: {stt_dir}")

if __name__ == "__main__":
    setup_stt_model("tiny")
    verify_models()
