from speech2text.config import AudioConfig, ModelConfig


def test_audio_config_defaults():
    config = AudioConfig()
    expected_sample_rate = 16000
    assert config.sample_rate == expected_sample_rate
    assert config.channels == 1


def test_model_config_defaults():
    config = ModelConfig()
    assert config.whisper_model == "medium"
    assert config.english_only is False
