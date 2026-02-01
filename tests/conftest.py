"""Pytest configuration and fixtures.

This module sets up mocks for GUI-dependent libraries that are not available
in headless CI environments.
"""

import sys
from unittest.mock import MagicMock

# Mock pynput before any imports that might use it
sys.modules["pynput"] = MagicMock()
sys.modules["pynput.keyboard"] = MagicMock()
sys.modules["pynput.mouse"] = MagicMock()

# Mock whisper_mic to prevent it from importing pynput
sys.modules["whisper_mic"] = MagicMock()
sys.modules["whisper_mic.whisper_mic"] = MagicMock()

# Mock sounddevice as it requires PortAudio
sys.modules["sounddevice"] = MagicMock()
