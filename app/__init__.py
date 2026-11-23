"""Deepgram STT Backend Application Package"""
import sys
import multiprocessing

# Ensure multiprocessing spawned worker processes on Windows use the virtualenv Python
if sys.platform == "win32":
    try:
        multiprocessing.set_executable(sys.executable)
    except Exception:
        pass

