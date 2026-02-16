# Troubleshooting

Here you will find solutions to common problems.

## CUDA Out of Memory

**Problem**: `torch.OutOfMemoryError` when loading the model.

**Solutions**:
1. Use a smaller model: `--model-size base`.
2. Use CPU instead of GPU: `--device cpu`.
3. Ensure no other processes are occupying VRAM.

## No Audio Detected

**Problem**: Recording stops immediately or no text is generated.

**Solutions**:
1. Check microphone permissions.
2. Adjust the `silence_threshold` in the `AudioConfig`.
3. Test your microphone with other applications.

## Installation Error with `sounddevice`

**Problem**: Error regarding `PortAudio`.

**Solutions**:
- **Ubuntu/Debian**: `sudo apt-get install libportaudio2`
- **macOS**: `brew install portaudio`
