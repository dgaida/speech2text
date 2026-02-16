# Docstring-Stilhandbuch

Wir folgen dem **Google-Stil** für alle Docstrings im Projekt. Dies gewährleistet Konsistenz und eine saubere Generierung der API-Dokumentation.

## Beispiel

```python
def function(arg1, arg2):
    """Kurze Zusammenfassung.

    Längere Beschreibung, falls erforderlich.

    Args:
        arg1 (int): Beschreibung des ersten Arguments.
        arg2 (str): Beschreibung des zweiten Arguments.

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wenn arg1 ungültig ist.
    """
```

## Anforderungen

- Jede öffentliche Klasse, Methode und Funktion muss einen Docstring haben.
- Typen sollten im Docstring angegeben werden, auch wenn Typ-Hints verwendet werden.
- Die Abdeckung wird durch `interrogate` erzwungen (Schwellenwert 95%).
