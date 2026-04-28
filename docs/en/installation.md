# Installation

## Prerequisites

- **Python**: 3.9 or higher.  
- **System Libraries**:  
    - **Linux**: `libportaudio2`, `libasound2-dev`.  
    - **macOS**: `portaudio`.  
    - **Windows**: Should work out of the box.  

## Install from Source

```bash
git clone https://github.com/dgaida/speech2text.git
cd speech2text
pip install .
```

## Development Installation

If you want to work on the documentation, install the additional dependencies:

```bash
pip install -e ".[docs,dev]"
```

## GPU Support (Optional)

To use an NVIDIA GPU, ensure you have the appropriate PyTorch version with CUDA support installed:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
