import os
from unittest.mock import patch

from speech2text.utils.file_utils import temporary_audio_file


def test_temporary_audio_file():
    with temporary_audio_file() as temp_path:
        assert os.path.exists(temp_path)
        assert temp_path.endswith(".wav")
        # Write something to it
        with open(temp_path, "w") as f:
            f.write("test")

    assert not os.path.exists(temp_path)


def test_temporary_audio_file_cleanup_error():
    with patch("os.remove") as mock_remove:
        mock_remove.side_effect = OSError("failed to remove")
        with temporary_audio_file() as temp_path:
            assert os.path.exists(temp_path)
        # Should hit line 31: logger.warning
