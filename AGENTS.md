# Agent Instructions for Speech2Text

Welcome, fellow agent! Here are some project-specific guidelines and instructions to help you work effectively on the `speech2text` repository.

## Project Overview
This project provides speech-to-text capabilities using OpenAI's Whisper model. It follows a Strategy Pattern for transcription backends and uses MkDocs with the Material theme for bilingual documentation.

## Coding Standards
- **Formatting:** Use `black` with a line length of 127.
- **Linting:** Use `ruff` for linting.
- **Type Checking:** Strict `mypy` type checking is enforced.
- **Documentation:** Public APIs must have Google-style docstrings. Docstring coverage is monitored with `interrogate`.

## Documentation
- **Multilingual:** Documentation is bilingual (German default, English secondary) using `mkdocs-static-i18n`.
- **Versioning:** Documentation is versioned using `mike`.
- **Configuration:** Always ensure `site_url` is set in `mkdocs.yml` and `mike set-default` is used in the deployment workflow to avoid 404 errors at the root URL.

## Testing
- **Suite:** Use `pytest` for unit and integration tests.
- **Coverage:** Aim for at least 80% test coverage.
- **Headless Environments:** If working in a headless environment (like most CI/CD pipelines), ensure that hardware-dependent libraries like `pynput`, `sounddevice`, and `pyaudio` are mocked in `tests/conftest.py`.

## Mandatory Cleanup
- **Temporary Files:** Always delete any temporary files, logs, or patch files created during your session before submitting a pull request or finishing your task. Examples include `code_review_request.txt`, `*.patch`, or temporary test artifacts.

## CI/CD
- **Workflows:** Be mindful of the workflows in `.github/workflows/`.
- **Automated Versioning:** The project uses automated versioning. Commits that shouldn't trigger CI (like version bumps) should include `[skip ci]`.
