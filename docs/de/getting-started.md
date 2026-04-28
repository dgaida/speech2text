# Erste Schritte

In diesem Abschnitt erfahren Sie, wie Sie schnell mit `speech2text` starten können.

## Schnelleinrichtung

1. Installieren Sie das Paket (siehe [Installation](installation.md)).  
2. Initialisieren Sie die `Speech2Text`-Klasse.  
3. Starten Sie die Aufnahme und Transkription.  

## Minimales Beispiel

```python
from speech2text import Speech2Text
import torch

# Empfohlene Verwendung mit Kontextmanager
with Speech2Text(
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_whisper_mic=True
) as stt:
    text = stt.record_and_transcribe()
    print(f"Transkribierter Text: {text}")
```

## Nächste Schritte

- Lesen Sie mehr über die [Konfiguration](configuration.md).  
- Entdecken Sie fortgeschrittene [Nutzungsszenarien](usage.md).  
- Schauen Sie sich die [API-Referenz](api.md) an.  
