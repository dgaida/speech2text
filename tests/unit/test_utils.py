import os

from speech2text.utils.file_utils import temporary_audio_file


def test_temporary_audio_file():
    with temporary_audio_file() as temp_path:
        assert os.path.exists(temp_path)
        assert temp_path.endswith(".wav")
        # Write something to it
        with open(temp_path, "w") as f:
            f.write("test")

    assert not os.path.exists(temp_path)
