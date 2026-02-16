# Usage

## Python API

The primary usage is via the `Speech2Text` class.

### Real-time Mode (WhisperMic)

This mode is optimized for direct speech input.

```python
from speech2text import Speech2Text
import torch

stt = Speech2Text(use_whisper_mic=True, device="cuda", torch_dtype=torch.float16)
text = stt.record_and_transcribe()
```

### Local Mode

This mode uses the Hugging Face Pipeline and allows for finer control over recording.

```python
from speech2text import Speech2Text
import torch

stt = Speech2Text(use_whisper_mic=False, device="cpu", torch_dtype=torch.float32)
text = stt.record_and_transcribe()
```

## Command-Line Interface (CLI)

The module can be run directly from the terminal:

```bash
# By default with WhisperMic
python -m speech2text

# With local model and smaller model size
python -m speech2text --no-whisper-mic --model-size base

# Multiple recordings in a row
python -m speech2text --recordings 3
```

### CLI Options

| Option | Description |
|--------|--------------|
| `--device` | Device for inference ('cuda', 'cpu'). |
| `--no-whisper-mic` | Use local Whisper instead of WhisperMic. |
| `--model-size` | Whisper model size. |
| `--verbose` | Detailed log output. |
| `--recordings` | Number of recordings to perform. |
