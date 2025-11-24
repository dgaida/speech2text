# Speech2Text Module

A Python package for speech-to-text (ASR) capabilities using OpenAI's Whisper model with automatic silence detection.

## Badges

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
- **Automatic Silence Detection**: Intelligently stops recording when no speech is detected
- **Multi-Language Support**: Automatic detection and translation to English
- **GPU Acceleration**: Optional CUDA support for faster inference
- **Flexible Configuration**: Customizable silence thresholds, device selection, and data types
- **Comprehensive Testing**: Full unit test coverage with mocked dependencies

## Installation

### Prerequisites

- Python 3.8 or higher
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
python main.py

# Use local Whisper model instead
python main.py --no-whisper-mic

# Enable verbose output
python main.py --verbose

# Record multiple times
python main.py --recordings 3

# Force CPU usage
python main.py --device cpu

# Use float32 precision
python main.py --dtype float32
```

### Using the Python API

```python
from speech2text import Speech2Text
import torch

# Initialize with WhisperMic (recommended for real-time)
stt = Speech2Text(
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    use_whisper_mic=True,
    verbose=True,
)

# Record and transcribe
text = stt.record_and_transcribe()
print("Transcribed text:", text)
```

## Usage

### Command-Line Interface

The `main.py` script provides a full-featured command-line interface:

```bash
python main.py [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--device` | `{cuda,cpu}` | auto-detect | Device for inference |
| `--no-whisper-mic` | flag | False | Use local Whisper instead of WhisperMic |
| `--verbose` | flag | False | Enable verbose output |
| `--recordings` | int | 1 | Number of recordings to perform |
| `--dtype` | `{float16,float32}` | float16 | Torch data type for inference |

#### Examples

```bash
# Single recording with GPU acceleration
python main.py --device cuda --verbose

# Multiple recordings with local Whisper model
python main.py --no-whisper-mic --recordings 5

# CPU-only with float32 precision
python main.py --device cpu --dtype float32
```

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
    verbose=False,
)

# Record and transcribe
transcription = stt.record_and_transcribe()
```

#### Advanced Configuration

```python
# Using local Whisper model with custom settings
stt = Speech2Text(
    device="cpu",
    torch_dtype=torch.float32,
    use_whisper_mic=False,
    verbose=True,
)

# Check verbosity setting
if stt.verbose():
    print("Verbose mode enabled")

# Perform transcription
result = stt.record_and_transcribe()
```

#### Custom Silence Detection

```python
# The _record_audio_until_silence method can be customized
# (Note: This is a private method, shown for reference)
audio_data, sample_rate = Speech2Text._record_audio_until_silence(
    silence_threshold=0.001,  # Amplitude threshold
    silence_duration=2.0,      # Seconds of silence before stopping
)
```

## Architecture

### File Structure

```
speech2text/
├── speech2text/
│   ├── __init__.py                # Package initialization
│   └── speech2text.py             # Main Speech2Text class
├── main.py                        # CLI application
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Project configuration
├── README.md                      # Documentation
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_speech2text.py       # Unit tests for Speech2Text
│   └── test_main.py               # Unit tests for CLI
└── .github/
    └── workflows/                 # CI/CD pipelines
        ├── tests.yml
        ├── lint.yml
        ├── codeql.yml
        └── release.yml
```

### Class: `Speech2Text`

#### Constructor

```python
Speech2Text(
    device: str,
    torch_dtype: type,
    use_whisper_mic: bool = True,
    verbose: bool = False
) -> None
```

**Parameters:**
- `device`: Device for inference (`"cuda"` or `"cpu"`)
- `torch_dtype`: Torch data type (`torch.float16` or `torch.float32`)
- `use_whisper_mic`: Whether to use WhisperMic (True) or local Whisper (False)
- `verbose`: Enable verbose logging

#### Methods

##### `record_and_transcribe() -> str`
Records speech from the microphone and returns the transcribed text.

**Returns:** Transcribed text as a string

**Raises:** Various exceptions based on the recording backend

##### `verbose() -> bool`
Returns the current verbosity setting.

**Returns:** `True` if verbose mode is enabled, `False` otherwise

#### Private Methods

- `_record_and_transcribe()`: Handles offline recording and transcription
- `_record_and_transcribe_whisper_mic()`: Handles real-time WhisperMic transcription
- `_record_audio_until_silence()`: Records audio until silence is detected

## Testing

The project includes comprehensive unit tests with high coverage.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=speech2text --cov-report=html

# Run specific test file
pytest tests/test_speech2text.py

# Run with verbose output
pytest -v
```

### Test Structure

- **`tests/test_speech2text.py`**: Tests for the Speech2Text class
  - Initialization tests (WhisperMic and local Whisper)
  - Recording and transcription tests
  - Silence detection tests
  - Edge cases and error handling

- **`tests/test_main.py`**: Tests for the CLI application
  - Argument parsing tests
  - Device selection tests
  - Main function workflow tests
  - Error handling tests

### Test Coverage

Current test coverage includes:
- ✅ Initialization with both backends
- ✅ Recording and transcription workflows
- ✅ Silence detection algorithm
- ✅ Command-line argument parsing
- ✅ Error handling and edge cases
- ✅ Verbose mode functionality

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/dgaida/speech2text.git
cd speech2text

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black ruff mypy bandit

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 127)
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy speech2text --ignore-missing-imports

# Security scan
bandit -r speech2text/
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically:

```bash
# Run manually on all files
pre-commit run --all-files
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Daniel Gaida**
Email: daniel.gaida@th-koeln.de
GitHub: [@dgaida](https://github.com/dgaida)

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- The [`whisper_mic`](https://github.com/mallorbc/whisper_mic) package maintainers

## Support

For issues, questions, or contributions, please:
- Open an issue on [GitHub Issues](https://github.com/dgaida/speech2text/issues)
- Submit a pull request for bug fixes or features
- Contact the author via email

---

**Note**: This module was developed as part of the `robot_environment` framework for multimodal robotic interaction.
