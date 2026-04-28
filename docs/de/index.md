# Speech2Text Übersicht

Willkommen zur Dokumentation von `speech2text`, einem leistungsstarken Python-Modul für die Spracherkennung (ASR) basierend auf OpenAIs Whisper-Modellen.

## Hauptmerkmale

- **Zwei Betriebsmodi**:  
    - **WhisperMic**: Echtzeit-Transkription direkt vom Mikrofon.  
    - **Lokales Whisper**: Offline-Transkription mit Hugging Face Pipelines.  
- **Intelligente Stille-Erkennung**: Stoppt die Aufnahme automatisch, wenn nicht mehr gesprochen wird.  
- **Strategie-Entwurfsmuster**: Flexibler Wechsel zwischen verschiedenen Transkriptions-Backends.  
- **GPU-Beschleunigung**: Unterstützung für CUDA für blitzschnelle Inferenzen.  
- **Kontextmanager**: Sichere Ressourcenverwaltung und Bereinigung.  

## Zielgruppe

Dieses Modul ist ideal für Entwickler, die Spracherkennungsfunktionen in ihre Python-Anwendungen integrieren möchten, sei es für Echtzeit-Assistenten oder Offline-Transkriptionsdienste.
