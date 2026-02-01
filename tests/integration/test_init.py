import pytest
import torch
from unittest.mock import patch
from speech2text.speech2text import Speech2Text
from speech2text.config import ModelConfig

@pytest.mark.integration
class TestSpeech2TextIntegration:
    """Integration tests that might require more resources."""

    @patch("speech2text.speech2text.pipeline")
    def test_initialization_real_pipeline_mocked(self, mock_pipeline):
        # We still mock the actual heavy model loading for unit-like integration test
        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=False,
            model_config=ModelConfig(whisper_model="tiny")
        )
        assert stt._asr_model is not None
