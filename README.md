# Speech2Text Module

A Python package for speech-to-text (ASR) capabilities using OpenAI's Whisper model with automatic silence detection.

## Badges

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/speech2text/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/speech2text)
[![Code Quality](https://github.com/dgaida/speech2text/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/speech2text/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/speech2text/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/speech2text/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/speech2text/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/speech2text/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Performance Guidelines](#performance-guidelines)
- [Architecture](#architecture)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

The `speech2text` module provides a Python interface to record audio from a microphone and convert it into text using automatic speech recognition (ASR). It features intelligent silence detection to automatically stop recording and supports both real-time and offline transcription modes.

## Features

- **Dual Operating Modes**:
  - **WhisperMic Mode**: Real-time recording and transcription using the [`whisper_mic`](https://github.com/mallorbc/whisper_mic) package
  - **Local Whisper Mode**: Offline transcription using Hugging Face's Whisper model pipeline
- **Strategy Pattern Architecture**: Easily switchable transcription backends
- **Automatic Silence Detection**: Intelligently stops recording when no speech is detected
- **Multi-Language Support**: Automatic detection and translation to English
- **GPU Acceleration**: Optional CUDA support for faster inference
- **Context Manager Support**: Automatic resource cleanup
- **Comprehensive Testing**: Unit and integration tests with high coverage

## Installation

### Prerequisites

- Python 3.9 or higher
- (Optional) CUDA-capable GPU for accelerated inference

### Install from Source

```bash
# Clone the repository
git clone https://github.com/dgaida/speech2text.git
cd speech2text

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Using the Command-Line Interface

```bash
# Basic usage (uses WhisperMic by default)
python -m speech2text

# Use local Whisper model instead
python -m speech2text --no-whisper-mic

# Specify model size
python -m speech2text --model-size small

# Enable verbose output
python -m speech2text --verbose

# Record multiple times
python -m speech2text --recordings 3
```

### Using the Python API

```python
from speech2text import Speech2Text
import torch

# Initialize with context manager (recommended for automatic cleanup)
with Speech2Text(
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    use_whisper_mic=True,
    verbose=True,
) as stt:
    # Record and transcribe
    text = stt.record_and_transcribe()
    print("Transcribed text:", text)
```

## Usage

### Command-Line Interface

The package can be run directly using `python -m speech2text`:

```bash
python -m speech2text [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--device` | string | auto-detect | Device for inference (e.g., 'cuda', 'cpu') |
| `--no-whisper-mic` | flag | False | Use local Whisper instead of WhisperMic |
| `--model-size` | `{tiny,base,small,medium,large}` | medium | Whisper model size |
| `--verbose` | flag | False | Enable verbose output |
| `--recordings` | int | 1 | Number of recordings to perform |
| `--dtype` | `{float16,float32}` | float16 | Torch data type for inference |

### Python API

#### Basic Usage

```python
from speech2text import Speech2Text
import torch

# Initialize
stt = Speech2Text(
    device="cuda",
    torch_dtype=torch.float16,
    use_whisper_mic=True,
)

try:
    # Record and transcribe
    transcription = stt.record_and_transcribe()
finally:
    # Always cleanup resources
    stt.cleanup()
```

#### Advanced Configuration

```python
from speech2text import Speech2Text, AudioConfig, ModelConfig
import torch

audio_cfg = AudioConfig(silence_threshold=0.001, silence_duration=2.0)
model_cfg = ModelConfig(whisper_model="small")

with Speech2Text(
    device="cpu",
    torch_dtype=torch.float32,
    use_whisper_mic=False,
    audio_config=audio_cfg,
    model_config=model_cfg,
    verbose=True,
) as stt:
    result = stt.record_and_transcribe()
```

## Troubleshooting

### CUDA Out of Memory
**Problem:** `torch.OutOfMemoryError` when initializing WhisperMic or the local model.

**Solutions:**
1. Use CPU instead: `--device cpu`
2. Use a smaller model: `--model-size small` or `--model-size base`
3. Use `float32` on CPU or ensure you have enough VRAM for `float16` on GPU.

### No Audio Detected
**Problem:** Recording stops immediately without transcription or fails to detect speech.

**Solutions:**
1. Check microphone permissions and ensure it's the default input device.
2. Adjust silence threshold: Initialize `AudioConfig` with a lower `silence_threshold`.
3. Check microphone input level in your OS settings.

### Installation Issues
**Problem:** Errors related to `sounddevice` or `portaudio`.

**Solutions:**
- **Ubuntu/Debian:** `sudo apt-get install libportaudio2`
- **macOS:** `brew install portaudio`
- **Windows:** Should work out of the box with the provided wheels.

## Performance Guidelines

| Model Size | VRAM Required | Relative Speed | Accuracy |
|------------|---------------|----------------|----------|
| tiny       | ~1 GB         | 32x            | Good     |
| base       | ~1 GB         | 16x            | Better   |
| small      | ~2 GB         | 6x             | Good     |
| medium     | ~5 GB         | 2x             | Better   |
| large      | ~10 GB        | 1x             | Best     |

**Recommendation:** Use `medium` model with `float16` on GPU for the best balance of speed and accuracy. Use `base` or `small` for low-resource environments.

## Architecture

### File Structure

```
speech2text/
├── speech2text/
│   ├── strategies/             # Transcription strategies (Strategy Pattern)
│   ├── utils/                  # Logging and file utilities
│   ├── __init__.py             # Package initialization
│   ├── __main__.py            # CLI entry point
│   ├── config.py               # Configuration dataclasses
│   ├── exceptions.py           # Custom exceptions
│   └── speech2text.py          # Main Speech2Text class
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures (audio files)
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # Documentation
```

## Testing

Run tests using `pytest`:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=speech2text
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development instructions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Daniel Gaida**  
Email: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)
