# Release-Prozess

## Versionsverwaltung

Wir verwenden [mike](https://github.com/jimporter/mike) für die Versionierung der Dokumentation.

## Schritte für ein neues Release

1. **Tag erstellen**: Erstellen Sie einen neuen Git-Tag (z.B. `v1.0.0`).  
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
2. **CI/CD**: Der GitHub Actions Workflow erkennt den Tag und:  
   - Baut die Dokumentation für diese Version.  
   - Veröffentlicht sie auf GitHub Pages.  
   - Aktualisiert den `latest`-Alias.  
3. **Changelog**: `git-cliff` generiert automatisch den Changelog basierend auf den Commits.  
