# Configuration

`speech2text` provides flexible configuration options via dataclasses.

## Audio Configuration

Use `AudioConfig` to control recording parameters.

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `sample_rate` | `int` | `16000` | Sample rate in Hz. |
| `chunk_duration` | `float` | `0.1` | Duration of each audio chunk in seconds. |
| `silence_threshold` | `float` | `0.0005` | Amplitude threshold for silence detection. |
| `silence_duration` | `float` | `3.0` | Duration of silence before stopping. |
| `channels` | `int` | `1` | Number of audio channels. |

## Model Configuration

Use `ModelConfig` to control Whisper model parameters.

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `whisper_model` | `str` | `medium` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). |
| `english_only` | `bool` | `False` | Use English-only model. |
| `pause_duration` | `int` | `1` | Pause duration for WhisperMic in seconds. |

## Advanced Configuration Example

```python
from speech2text import Speech2Text, AudioConfig, ModelConfig

audio_cfg = AudioConfig(silence_threshold=0.001, silence_duration=2.0)
model_cfg = ModelConfig(whisper_model="small")

with Speech2Text(
    device="cpu",
    audio_config=audio_cfg,
    model_config=model_cfg
) as stt:
    text = stt.record_and_transcribe()
```
