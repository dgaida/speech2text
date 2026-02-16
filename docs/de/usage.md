# Nutzung

## Python API

Die primäre Nutzung erfolgt über die `Speech2Text`-Klasse.

### Echtzeit-Modus (WhisperMic)

Dieser Modus ist für die direkte Spracheingabe optimiert.

```python
from speech2text import Speech2Text
import torch

stt = Speech2Text(use_whisper_mic=True, device="cuda", torch_dtype=torch.float16)
text = stt.record_and_transcribe()
```

### Lokaler Modus

Dieser Modus nutzt die Hugging Face Pipeline und erlaubt eine feinere Kontrolle über die Aufnahme.

```python
from speech2text import Speech2Text
import torch

stt = Speech2Text(use_whisper_mic=False, device="cpu", torch_dtype=torch.float32)
text = stt.record_and_transcribe()
```

## Kommandozeile (CLI)

Das Modul kann direkt aus dem Terminal ausgeführt werden:

```bash
# Standardmäßig mit WhisperMic
python -m speech2text

# Mit lokalem Modell und kleinerer Modellgröße
python -m speech2text --no-whisper-mic --model-size base

# Mehrere Aufnahmen hintereinander
python -m speech2text --recordings 3
```

### CLI-Optionen

| Option | Beschreibung |
|--------|--------------|
| `--device` | Gerät für die Inferenz ('cuda', 'cpu'). |
| `--no-whisper-mic` | Lokales Whisper statt WhisperMic verwenden. |
| `--model-size` | Whisper-Modellgröße. |
| `--verbose` | Detaillierte Log-Ausgaben. |
| `--recordings` | Anzahl der durchzuführenden Aufnahmen. |
