"""Local Audio Processing Package

Provides format validation, decoding, resampling, mono conversion, 
VAD/silence trimming, duration calculation, and temporary file lifecycle management.
"""

from .processor import AudioProcessor

__all__ = ["AudioProcessor"]
