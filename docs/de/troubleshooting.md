# Fehlerbehebung

Hier finden Sie Lösungen für häufig auftretende Probleme.

## CUDA Out of Memory

**Problem**: `torch.OutOfMemoryError` beim Laden des Modells.

**Lösungen**:
1. Verwenden Sie ein kleineres Modell: `--model-size base`.
2. Nutzen Sie die CPU statt der GPU: `--device cpu`.
3. Stellen Sie sicher, dass keine anderen Prozesse den VRAM belegen.

## Kein Audio erkannt

**Problem**: Die Aufnahme stoppt sofort oder es wird kein Text generiert.

**Lösungen**:
1. Prüfen Sie die Mikrofon-Berechtigungen.
2. Passen Sie die `silence_threshold` in der `AudioConfig` an.
3. Testen Sie Ihr Mikrofon mit anderen Anwendungen.

## Installationsfehler bei `sounddevice`

**Problem**: Fehler bezüglich `PortAudio`.

**Lösungen**:
- **Ubuntu/Debian**: `sudo apt-get install libportaudio2`
- **macOS**: `brew install portaudio`
