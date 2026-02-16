"""Configuration constants for speech2text module."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AudioConfig:
    """Audio recording configuration.

    Attributes:
        sample_rate (int): Sample rate for audio recording in Hz.
        chunk_duration (float): Duration of each audio chunk in seconds.
        silence_threshold (float): Amplitude threshold to detect silence.
        silence_duration (float): Duration of continuous silence before stopping.
        channels (int): Number of audio channels (1 = mono).
    """

    sample_rate: int = 16000
    chunk_duration: float = 0.1
    silence_threshold: float = 0.0005
    silence_duration: float = 3.0
    channels: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration.

    Attributes:
        whisper_model (str): Whisper model size to use ("tiny", "base", "small", "medium", "large").
        english_only (bool): Whether to use English-only model.
        pause_duration (int): Pause duration for WhisperMic in seconds.
    """

    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "medium"
    english_only: bool = False
    pause_duration: int = 1


DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
