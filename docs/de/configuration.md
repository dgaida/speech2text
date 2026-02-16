# Konfiguration

`speech2text` bietet flexible Konfigurationsoptionen über Dataklassen.

## Audio-Konfiguration

Verwenden Sie `AudioConfig` zur Steuerung der Aufnahme-Parameter.

| Parameter | Typ | Standard | Beschreibung |
|-----------|------|----------|-------------|
| `sample_rate` | `int` | `16000` | Abtastrate in Hz. |
| `chunk_duration` | `float` | `0.1` | Dauer jedes Audio-Chunks in Sekunden. |
| `silence_threshold` | `float` | `0.0005` | Amplitudenschwelle zur Stille-Erkennung. |
| `silence_duration` | `float` | `3.0` | Dauer der Stille vor dem Stoppen. |
| `channels` | `int` | `1` | Anzahl der Audio-Kanäle. |

## Modell-Konfiguration

Verwenden Sie `ModelConfig` zur Steuerung der Whisper-Modellparameter.

| Parameter | Typ | Standard | Beschreibung |
|-----------|------|----------|-------------|
| `whisper_model` | `str` | `medium` | Größe des Whisper-Modells (`tiny`, `base`, `small`, `medium`, `large`). |
| `english_only` | `bool` | `False` | Nur englisches Modell verwenden. |
| `pause_duration` | `int` | `1` | Pausendauer für WhisperMic in Sekunden. |

## Beispiel für fortgeschrittene Konfiguration

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
