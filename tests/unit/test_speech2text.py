import pytest
from unittest.mock import MagicMock, patch
import torch
from speech2text.speech2text import Speech2Text

class TestSpeech2Text:
    @patch("speech2text.speech2text.WhisperMic")
    @patch("speech2text.speech2text.WhisperMicStrategy")
    def test_init_with_whisper_mic(self, mock_strategy, mock_whisper_mic):
        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False
        )
        assert stt._whisper_mic is not None
        mock_whisper_mic.assert_called_once()
        mock_strategy.assert_called_once()

    @patch("speech2text.speech2text.pipeline")
    @patch("speech2text.speech2text.LocalWhisperStrategy")
    def test_init_without_whisper_mic(self, mock_strategy, mock_pipeline):
        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=False,
            verbose=False
        )
        assert stt._asr_model is not None
        mock_pipeline.assert_called_once()
        mock_strategy.assert_called_once()

    def test_invalid_device(self):
        with pytest.raises(ValueError):
            Speech2Text(device="invalid", torch_dtype=torch.float32)

    @patch("speech2text.speech2text.WhisperMic")
    def test_cleanup(self, mock_whisper_mic):
        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=True)
        stt.cleanup()
        assert stt._whisper_mic is None

    @patch("speech2text.speech2text.WhisperMic")
    def test_context_manager(self, mock_whisper_mic):
        with Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=True) as stt:
            assert stt._whisper_mic is not None
        assert stt._whisper_mic is None

    @patch("speech2text.speech2text.WhisperMic")
    def test_record_and_transcribe_delegation(self, mock_whisper_mic):
        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=True)
        stt._strategy = MagicMock()
        stt._strategy.transcribe.return_value = "delegated"
        assert stt.record_and_transcribe() == "delegated"
        stt._strategy.transcribe.assert_called_once()

    @patch("speech2text.speech2text.WhisperMic")
    def test_verbose_getter(self, mock_whisper_mic):
        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, verbose=True)
        assert stt.verbose() is True
