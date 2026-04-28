# Testing

The project includes comprehensive unit tests with high coverage.

## Running Tests

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

## Test Structure

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

## Test Coverage

Current test coverage includes:  
- ✅ Initialization with both backends  
- ✅ Recording and transcription workflows  
- ✅ Silence detection algorithm  
- ✅ Command-line argument parsing  
- ✅ Error handling and edge cases  
- ✅ Verbose mode functionality  
