import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def temporary_audio_file(suffix: str = ".wav") -> Generator[str, None, None]:
    """Context manager for safe temporary audio file handling.

    Args:
        suffix: File extension for the temporary file.

    Yields:
        str: Path to the temporary file.
    """
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="speech2text_")
        os.close(fd)  # Close file descriptor, we only need the path
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")
