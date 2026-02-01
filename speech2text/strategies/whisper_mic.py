from whisper_mic import WhisperMic

from speech2text.exceptions import TranscriptionError
from speech2text.strategies.base import TranscriptionStrategy
from speech2text.utils.logging import get_logger

logger = get_logger(__name__)


class WhisperMicStrategy(TranscriptionStrategy):
    """Transcription strategy using WhisperMic."""

    def __init__(self, whisper_mic: WhisperMic, verbose: bool = False):
        """Initialize WhisperMic strategy.

        Args:
            whisper_mic: Initialized WhisperMic instance.
            verbose: Enable verbose output.
        """
        self._whisper_mic = whisper_mic
        self._verbose = verbose

    def transcribe(self) -> str:
        """Record and transcribe using WhisperMic.

        Returns:
            str: Transcribed text.
        """
        try:
            result = self._whisper_mic.listen()
            if self._verbose:
                logger.info(f"Transcription result: {result}")
            return str(result)
        except Exception as e:
            raise TranscriptionError(f"WhisperMic transcription failed: {e}") from e
