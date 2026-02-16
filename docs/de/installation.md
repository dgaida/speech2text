# Installation

## Voraussetzungen

- **Python**: 3.9 oder höher.
- **System-Bibliotheken**:
    - **Linux**: `libportaudio2`, `libasound2-dev`.
    - **macOS**: `portaudio`.
    - **Windows**: Sollte standardmäßig funktionieren.

## Installation aus dem Quellcode

```bash
git clone https://github.com/dgaida/speech2text.git
cd speech2text
pip install .
```

## Entwicklungs-Installation

Wenn Sie an der Dokumentation arbeiten möchten, installieren Sie die zusätzlichen Abhängigkeiten:

```bash
pip install -e ".[docs,dev]"
```

## GPU-Unterstützung (Optional)

Um eine NVIDIA GPU zu verwenden, stellen Sie sicher, dass Sie die passende PyTorch-Version mit CUDA-Unterstützung installiert haben:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
