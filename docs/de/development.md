# Entwicklung

## Einrichten der Umgebung

Folgen Sie den Anweisungen in der [Installation](installation.md) für die Entwicklungs-Umgebung.

## Kodierrichtlinien

- **Docstrings**: Alle öffentlichen APIs müssen Google-Style Docstrings haben.  
- **Formatierung**: Wir verwenden `black` mit einer Zeilenlänge von 127.  
- **Linting**: Wir verwenden `ruff`.  
- **Typisierung**: Strenge Typisierung mit `mypy` ist erforderlich.  

## Tests ausführen

Wir verwenden `pytest` für unsere Testsuite.

```bash
# Alle Tests ausführen
pytest

# Mit Coverage-Bericht
pytest --cov=speech2text
```

## Dokumentation lokal bauen

```bash
# Live-Vorschau
mkdocs serve

# Statische Seite bauen
mkdocs build
```
