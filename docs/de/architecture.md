# Architektur

## Systemübersicht

`speech2text` nutzt das **Strategy Pattern**, um zwischen verschiedenen Transkriptions-Backends zu wechseln.

```mermaid
graph TD
    A[Nutzer/CLI] --> B[Speech2Text]
    B --> C{TranscriptionStrategy}
    C --> D[WhisperMicStrategy]
    C --> E[LocalWhisperStrategy]
    D --> F[WhisperMic Package]
    E --> G[Hugging Face Pipeline]
    G --> H[OpenAI Whisper Model]
```

## Datenfluss

1.  **Initialisierung**: Das gewünschte Modell und die Strategie werden geladen.  
2.  **Aufnahme**: Audio wird vom Mikrofon gestreamt.  
3.  **Stille-Erkennung**: Das System analysiert die Amplitude und stoppt nach einer definierten Ruhezeit.  
4.  **Verarbeitung**: Die Audio-Daten werden an das Whisper-Modell gesendet.  
5.  **Ausgabe**: Der transkribierte Text wird zurückgegeben.  

```mermaid
sequenceDiagram
    participant U as Nutzer
    participant S as Speech2Text
    participant ST as Strategy
    participant M as Modell

    U->>S: record_and_transcribe()
    S->>ST: transcribe()
    ST->>ST: Aufnahme starten
    Note over ST: Audio-Puffer füllen
    ST->>ST: Stille erkannt?
    ST->>M: Audio zur Inferenz senden
    M-->>ST: Text-Ergebnis
    ST-->>S: Transkription
    S-->>U: Finaler Text
```
