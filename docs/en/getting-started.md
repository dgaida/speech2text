# Getting Started

In this section, you will learn how to quickly get started with `speech2text`.

## Quick Setup

1. Install the package (see [Installation](installation.md)).
2. Initialize the `Speech2Text` class.
3. Start recording and transcription.

## Minimal Example

```python
from speech2text import Speech2Text
import torch

# Recommended usage with context manager
with Speech2Text(
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_whisper_mic=True
) as stt:
    text = stt.record_and_transcribe()
    print(f"Transcribed text: {text}")
```

## Next Steps

- Read more about [Configuration](configuration.md).
- Explore advanced [Usage scenarios](usage.md).
- Check out the [API Reference](api.md).
