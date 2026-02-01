from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech2text.exceptions import TranscriptionError
from speech2text.strategies.local_whisper import LocalWhisperStrategy
from speech2text.strategies.whisper_mic import WhisperMicStrategy


class TestLocalWhisperStrategy:
    @patch("speech2text.strategies.local_whisper.sd.InputStream")
    @patch("speech2text.strategies.local_whisper.write")
    @patch("speech2text.strategies.local_whisper.temporary_audio_file")
    def test_transcribe(self, mock_temp, mock_write, mock_stream):
        mock_asr = MagicMock()
        mock_asr.return_value = {"text": "hello"}

        mock_temp.return_value.__enter__.return_value = "temp.wav"

        # Mock record_audio_until_silence internal part
        mock_stream_instance = MagicMock()
        mock_stream_instance.read.side_effect = [
            (np.zeros((1600, 1)), None),  # Silence
            (np.zeros((1600, 1)), None),
            (np.zeros((1600, 1)), None),
            (np.zeros((1600, 1)), None),
        ]
        mock_stream.return_value.__enter__.return_value = mock_stream_instance

        # We need to mock time.time to trigger silence detection quickly
        with patch("speech2text.strategies.local_whisper.time.time") as mock_time:
            mock_time.side_effect = [0, 0.1, 0.2, 5.0]  # 5.0 - 0.2 > silence_duration (3.0)

            strategy = LocalWhisperStrategy(asr_model=mock_asr)
            result = strategy.transcribe()

            assert result == "hello"
            mock_asr.assert_called_once()


class TestWhisperMicStrategy:
    def test_transcribe(self):
        mock_mic = MagicMock()
        mock_mic.listen.return_value = "world"

        strategy = WhisperMicStrategy(whisper_mic=mock_mic)
        result = strategy.transcribe()

        assert result == "world"
        mock_mic.listen.assert_called_once()

    def test_transcribe_failure(self):
        mock_mic = MagicMock()
        mock_mic.listen.side_effect = Exception("error")

        strategy = WhisperMicStrategy(whisper_mic=mock_mic)
        with pytest.raises(TranscriptionError):
            strategy.transcribe()
