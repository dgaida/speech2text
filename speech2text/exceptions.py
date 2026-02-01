"""Custom exceptions for speech2text module."""


class Speech2TextError(Exception):
    """Base exception for speech2text module."""

    pass


class RecordingError(Speech2TextError):
    """Error during audio recording."""

    pass


class TranscriptionError(Speech2TextError):
    """Error during transcription."""

    pass


class ConfigurationError(Speech2TextError):
    """Error in configuration."""

    pass
