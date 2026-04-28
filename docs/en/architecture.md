# Architecture

## System Overview

`speech2text` uses the **Strategy Pattern** to switch between different transcription backends.

```mermaid
graph TD
    A[User/CLI] --> B[Speech2Text]
    B --> C{TranscriptionStrategy}
    C --> D[WhisperMicStrategy]
    C --> E[LocalWhisperStrategy]
    D --> F[WhisperMic Package]
    E --> G[Hugging Face Pipeline]
    G --> H[OpenAI Whisper Model]
```

## Data Flow

1.  **Initialization**: The desired model and strategy are loaded.  
2.  **Recording**: Audio is streamed from the microphone.  
3.  **Silence Detection**: The system analyzes the amplitude and stops after a defined period of silence.  
4.  **Processing**: The audio data is sent to the Whisper model.  
5.  **Output**: The transcribed text is returned.  

```mermaid
sequenceDiagram
    participant U as User
    participant S as Speech2Text
    participant ST as Strategy
    participant M as Model

    U->>S: record_and_transcribe()
    S->>ST: transcribe()
    ST->>ST: Start recording
    Note over ST: Fill audio buffer
    ST->>ST: Silence detected?
    ST->>M: Send audio for inference
    M-->>ST: Text result
    ST-->>S: Transcription
    S-->>U: Final text
```
