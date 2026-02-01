from typing import Protocol, runtime_checkable


@runtime_checkable
class TranscriptionStrategy(Protocol):
    """Protocol for transcription strategies."""

    def transcribe(self) -> str:
        """Record and transcribe audio.

        Returns:
            str: Transcribed text.

        Raises:
            RecordingError: If recording fails.
            TranscriptionError: If transcription fails.
        """
        ...  # pragma: no cover
