from speech2text.config import AudioConfig, ModelConfig

def test_audio_config_defaults():
    config = AudioConfig()
    assert config.sample_rate == 16000
    assert config.channels == 1

def test_model_config_defaults():
    config = ModelConfig()
    assert config.whisper_model == "medium"
    assert config.english_only is False
