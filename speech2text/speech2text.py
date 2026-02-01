"""Speech-to-text module with automatic speech recognition (ASR) capabilities."""

from typing import Any, Optional, Union

import torch
from transformers import Pipeline, pipeline
from whisper_mic import WhisperMic

from speech2text.config import DEFAULT_AUDIO_CONFIG, DEFAULT_MODEL_CONFIG, AudioConfig, ModelConfig
from speech2text.strategies.base import TranscriptionStrategy
from speech2text.strategies.local_whisper import LocalWhisperStrategy
from speech2text.strategies.whisper_mic import WhisperMicStrategy
from speech2text.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class Speech2Text:
    """Speech-to-text class with automatic speech recognition (ASR) capabilities.

    This class supports two ASR modes:
    1. Using the `whisper_mic` package for live transcription via microphone.
    2. Using the Hugging Face Whisper model for local recording and transcription.

    Attributes:
        _verbose: Whether verbose logging is enabled.
        _asr_model: The Whisper ASR model loaded via Hugging Face (optional).
        _whisper_mic: The WhisperMic instance used for microphone-based ASR (optional).
        _strategy: The transcription strategy to use.
    """

    _verbose: bool
    _asr_model: Optional[Pipeline]
    _whisper_mic: Optional[WhisperMic]
    _strategy: TranscriptionStrategy

    def __init__(
        self,
        device: Union[str, torch.device],
        torch_dtype: torch.dtype,
        use_whisper_mic: bool = True,
        verbose: bool = False,
        audio_config: AudioConfig = DEFAULT_AUDIO_CONFIG,
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    ) -> None:
        """Initializes the Speech2Text class and loads the ASR model.

        Args:
            device: The device to run the ASR model on ("cpu", "cuda", or torch.device).
            torch_dtype: The torch data type (torch.float16, torch.float32, etc.).
            use_whisper_mic: Whether to use the `whisper_mic` package. Defaults to True.
            verbose: Whether to enable verbose output. Defaults to False.
            audio_config: Audio recording configuration.
            model_config: Model configuration.

        Raises:
            ValueError: If device or torch_dtype is invalid.
            RuntimeError: If model initialization fails.
        """
        self._verbose = verbose
        setup_logging(verbose)

        # Input validation
        if isinstance(device, str):
            if device not in ("cpu", "cuda"):
                # Check for specific indices like cuda:0
                if not device.startswith("cuda:"):
                    raise ValueError(f"Invalid device: {device}. Expected 'cpu' or 'cuda'.")
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA device requested but CUDA is not available. Falling back to CPU.")
                device = "cpu"

        if use_whisper_mic:
            try:
                self._whisper_mic = WhisperMic(
                    model=model_config.whisper_model,
                    english=model_config.english_only,
                    pause=model_config.pause_duration,
                    device=str(device),
                )
                self._strategy = WhisperMicStrategy(self._whisper_mic, verbose=verbose)
                self._asr_model = None
            except (AttributeError, AssertionError, torch.OutOfMemoryError) as e:
                logger.error(f"Failed to initialize WhisperMic: {e}")
                raise RuntimeError(f"WhisperMic initialization failed: {e}") from e
        else:
            try:
                self._asr_model = pipeline(
                    "automatic-speech-recognition",
                    model=f"openai/whisper-{model_config.whisper_model}",
                    torch_dtype=torch_dtype,
                    device=device,
                )
                self._strategy = LocalWhisperStrategy(self._asr_model, audio_config=audio_config, verbose=verbose)
                self._whisper_mic = None
            except Exception as e:
                logger.error(f"Failed to initialize local Whisper model: {e}")
                raise RuntimeError(f"Local Whisper model initialization failed: {e}") from e

    def record_and_transcribe(self) -> str:
        """Records speech and transcribes it using the selected strategy.

        Returns:
            str: The transcribed text.

        Raises:
            RecordingError: If recording fails.
            TranscriptionError: If transcription fails.
        """
        return self._strategy.transcribe()

    def verbose(self) -> bool:
        """Returns whether verbose mode is active.

        Returns:
            bool: True if verbose output is enabled, False otherwise.
        """
        return self._verbose

    def __enter__(self) -> "Speech2Text":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources held by the instance."""
        if self._whisper_mic is not None:
            # WhisperMic doesn't have an explicit cleanup method, but we can null it
            self._whisper_mic = None

        if self._asr_model is not None:
            # Move model to CPU and clear CUDA cache if possible
            if hasattr(self._asr_model, "model"):
                self._asr_model.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._asr_model = None

        logger.info("Speech2Text resources cleaned up.")
