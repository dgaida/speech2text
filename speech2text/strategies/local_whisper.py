import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Pipeline
from speech2text.strategies.base import TranscriptionStrategy
from speech2text.config import AudioConfig, DEFAULT_AUDIO_CONFIG
from speech2text.exceptions import RecordingError, TranscriptionError
from speech2text.utils.file_utils import temporary_audio_file
from speech2text.utils.logging import get_logger

logger = get_logger(__name__)

class LocalWhisperStrategy(TranscriptionStrategy):
    """Transcription strategy using local Whisper model."""

    def __init__(
        self,
        asr_model: Pipeline,
        audio_config: AudioConfig = DEFAULT_AUDIO_CONFIG,
        verbose: bool = False
    ):
        """Initialize local Whisper strategy.

        Args:
            asr_model: Hugging Face ASR pipeline.
            audio_config: Audio recording configuration.
            verbose: Enable verbose output.
        """
        self._asr_model = asr_model
        self._audio_config = audio_config
        self._verbose = verbose

    def transcribe(self) -> str:
        """Record and transcribe using local Whisper model.

        Returns:
            str: Transcribed text.

        Raises:
            RecordingError: If audio recording fails.
            TranscriptionError: If transcription fails.
        """
        try:
            audio_data, sample_rate = self._record_audio_until_silence()
        except Exception as e:
            raise RecordingError(f"Audio recording failed: {e}") from e

        try:
            with temporary_audio_file() as temp_path:
                write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))
                result = self._asr_model(
                    temp_path,
                    generate_kwargs={"task": "translate"}
                )
                return str(result["text"])
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _record_audio_until_silence(self) -> tuple[np.ndarray, int]:
        """Record audio until silence is detected."""
        sample_rate = self._audio_config.sample_rate
        audio_data = []
        silence_start = None

        if self._verbose:
            logger.info("Recording... Speak now.")

        try:
            with sd.InputStream(samplerate=sample_rate, channels=self._audio_config.channels) as stream:
                while True:
                    chunk, _ = stream.read(int(sample_rate * self._audio_config.chunk_duration))
                    audio_data.append(chunk)

                    volume = np.linalg.norm(chunk) / len(chunk)
                    # if self._verbose:
                    #     logger.debug(f"Volume: {volume}")

                    if volume < self._audio_config.silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= self._audio_config.silence_duration:
                            if self._verbose:
                                logger.info("Silence detected. Stopping recording.")
                            break
                    else:
                        silence_start = None
        except Exception as e:
            raise RecordingError(f"Failed to record audio: {e}") from e

        if not audio_data:
            raise RecordingError("No audio data recorded.")

        audio_data_np = np.concatenate(audio_data, axis=0)
        return audio_data_np, sample_rate
