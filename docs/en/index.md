# Speech2Text Overview

Welcome to the documentation for `speech2text`, a powerful Python module for speech recognition (ASR) based on OpenAI's Whisper models.

## Key Features

- **Dual Operating Modes**:
    - **WhisperMic**: Real-time transcription directly from the microphone.
    - **Local Whisper**: Offline transcription using Hugging Face pipelines.
- **Intelligent Silence Detection**: Automatically stops recording when speech ends.
- **Strategy Pattern Architecture**: Flexible switching between different transcription backends.
- **GPU Acceleration**: CUDA support for lightning-fast inference.
- **Context Manager**: Safe resource management and cleanup.

## Target Audience

This module is ideal for developers looking to integrate speech recognition capabilities into their Python applications, whether for real-time assistants or offline transcription services.
