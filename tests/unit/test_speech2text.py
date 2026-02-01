from unittest.mock import MagicMock, patch

import pytest
import torch

from speech2text.speech2text import Speech2Text


class TestSpeech2Text:
    @patch("speech2text.speech2text.WhisperMic")
    @patch("speech2text.speech2text.WhisperMicStrategy")
    def test_init_with_whisper_mic(self, mock_strategy, mock_whisper_mic):
        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=True, verbose=False)
        assert stt._whisper_mic is not None
        mock_whisper_mic.assert_called_once()
        mock_strategy.assert_called_once()

    @patch("speech2text.speech2text.pipeline")
    @patch("speech2text.speech2text.LocalWhisperStrategy")
    def test_init_without_whisper_mic(self, mock_strategy, mock_pipeline):
        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=False, verbose=False)
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

    @patch("speech2text.speech2text.torch.cuda.is_available")
    @patch("speech2text.speech2text.WhisperMic")
    def test_init_cuda_fallback(self, mock_whisper_mic, mock_cuda_available):
        mock_cuda_available.return_value = False
        stt = Speech2Text(device="cuda", torch_dtype=torch.float32, use_whisper_mic=True)
        assert stt._whisper_mic is not None
        # Verify mock order: mock_whisper_mic should be WhisperMic, mock_cuda_available should be is_available
        mock_whisper_mic.assert_called_once()
        mock_cuda_available.assert_called()

    @patch("speech2text.speech2text.WhisperMic")
    def test_init_whisper_mic_runtime_error(self, mock_whisper_mic):
        mock_whisper_mic.side_effect = torch.OutOfMemoryError("OOM")
        with pytest.raises(RuntimeError, match="WhisperMic initialization failed"):
            Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=True)

    @patch("speech2text.speech2text.pipeline")
    def test_init_local_whisper_runtime_error(self, mock_pipeline):
        mock_pipeline.side_effect = Exception("init failed")
        with pytest.raises(RuntimeError, match="Local Whisper model initialization failed"):
            Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=False)

    @patch("speech2text.speech2text.torch.cuda.is_available")
    @patch("speech2text.speech2text.torch.cuda.empty_cache")
    @patch("speech2text.speech2text.pipeline")
    def test_cleanup_full(self, mock_pipeline, mock_empty_cache, mock_cuda_available):
        mock_cuda_available.return_value = True
        mock_asr = MagicMock()
        mock_pipeline.return_value = mock_asr

        stt = Speech2Text(device="cpu", torch_dtype=torch.float32, use_whisper_mic=False)
        stt.cleanup()

        # Verify mocks were called correctly, confirming correct argument order
        mock_pipeline.assert_called_once()
        mock_asr.model.to.assert_called_with("cpu")
        mock_empty_cache.assert_called_once()
        mock_cuda_available.assert_called()
        assert stt._asr_model is None
