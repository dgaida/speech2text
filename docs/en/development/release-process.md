# Release Process

## Versioning

We use [mike](https://github.com/jimporter/mike) for documentation versioning.

## Steps for a New Release

1. **Create Tag**: Create a new Git tag (e.g., `v1.0.0`).
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
2. **CI/CD**: The GitHub Actions workflow detects the tag and:
   - Builds the documentation for this version.
   - Publishes it to GitHub Pages.
   - Updates the `latest` alias.
3. **Changelog**: `git-cliff` automatically generates the changelog based on the commits.
