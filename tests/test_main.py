"""Unit tests for the main.py application module.

This module tests command-line argument parsing, device selection,
and the main application flow.
"""

import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import torch

import main
from main import (
    get_device,
    get_torch_dtype,
    initialize_speech2text,
    parse_arguments,
    perform_recording,
)


class TestParseArguments(unittest.TestCase):
    """Test cases for command-line argument parsing."""

    def test_parse_arguments_defaults(self) -> None:
        """Test parsing with default arguments."""
        with patch("sys.argv", ["main.py"]):
            args = parse_arguments()

            self.assertIsNone(args.device)
            self.assertFalse(args.no_whisper_mic)
            self.assertFalse(args.verbose)
            self.assertEqual(args.recordings, 1)
            self.assertEqual(args.dtype, "float16")

    def test_parse_arguments_custom(self) -> None:
        """Test parsing with custom arguments."""
        test_args = [
            "main.py",
            "--device",
            "cpu",
            "--no-whisper-mic",
            "--verbose",
            "--recordings",
            "3",
            "--dtype",
            "float32",
        ]

        with patch("sys.argv", test_args):
            args = parse_arguments()

            self.assertEqual(args.device, "cpu")
            self.assertTrue(args.no_whisper_mic)
            self.assertTrue(args.verbose)
            self.assertEqual(args.recordings, 3)
            self.assertEqual(args.dtype, "float32")

    def test_parse_arguments_invalid_device(self) -> None:
        """Test parsing with invalid device argument."""
        with patch("sys.argv", ["main.py", "--device", "invalid"]):
            with self.assertRaises(SystemExit):
                parse_arguments()

    def test_parse_arguments_invalid_dtype(self) -> None:
        """Test parsing with invalid dtype argument."""
        with patch("sys.argv", ["main.py", "--dtype", "float64"]):
            with self.assertRaises(SystemExit):
                parse_arguments()


class TestGetDevice(unittest.TestCase):
    """Test cases for device selection."""

    def test_get_device_with_argument(self) -> None:
        """Test device selection with explicit argument."""
        self.assertEqual(get_device("cuda"), "cuda")
        self.assertEqual(get_device("cpu"), "cpu")

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cuda(self, mock_cuda: Mock) -> None:
        """Test automatic device selection with CUDA available.

        Args:
            mock_cuda: Mocked torch.cuda.is_available function.
        """
        mock_cuda.return_value = True

        device = get_device(None)

        self.assertEqual(device, "cuda")
        mock_cuda.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cpu(self, mock_cuda: Mock) -> None:
        """Test automatic device selection without CUDA.

        Args:
            mock_cuda: Mocked torch.cuda.is_available function.
        """
        mock_cuda.return_value = False

        device = get_device(None)

        self.assertEqual(device, "cpu")
        mock_cuda.assert_called_once()


class TestGetTorchDtype(unittest.TestCase):
    """Test cases for torch dtype conversion."""

    def test_get_torch_dtype_float16(self) -> None:
        """Test conversion of float16 string to torch.float16."""
        dtype = get_torch_dtype("float16")
        self.assertEqual(dtype, torch.float16)

    def test_get_torch_dtype_float32(self) -> None:
        """Test conversion of float32 string to torch.float32."""
        dtype = get_torch_dtype("float32")
        self.assertEqual(dtype, torch.float32)

    def test_get_torch_dtype_invalid(self) -> None:
        """Test conversion with invalid dtype string."""
        with self.assertRaises(ValueError) as context:
            get_torch_dtype("float64")

        self.assertIn("Unknown dtype", str(context.exception))


class TestInitializeSpeech2Text(unittest.TestCase):
    """Test cases for Speech2Text initialization."""

    @patch("main.Speech2Text")
    def test_initialize_speech2text_success(self, mock_speech2text: Mock) -> None:
        """Test successful Speech2Text initialization.

        Args:
            mock_speech2text: Mocked Speech2Text class.
        """
        mock_instance = MagicMock()
        mock_speech2text.return_value = mock_instance

        result = initialize_speech2text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

        self.assertEqual(result, mock_instance)
        mock_speech2text.assert_called_once_with(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

    @patch("main.Speech2Text")
    def test_initialize_speech2text_failure(self, mock_speech2text: Mock) -> None:
        """Test Speech2Text initialization failure.

        Args:
            mock_speech2text: Mocked Speech2Text class that raises exception.
        """
        mock_speech2text.side_effect = Exception("Initialization failed")

        result = initialize_speech2text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

        self.assertIsNone(result)


class TestPerformRecording(unittest.TestCase):
    """Test cases for recording execution."""

    def test_perform_recording_success(self) -> None:
        """Test successful recording and transcription."""
        mock_stt = MagicMock()
        mock_stt.record_and_transcribe.return_value = "Test transcription"

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            perform_recording(mock_stt, 1, 1)

            output = captured_output.getvalue()

            # Verify output contains expected elements
            self.assertIn("Recording 1/1", output)
            self.assertIn("Test transcription", output)
            self.assertIn("Processing time", output)

            # Verify method was called
            mock_stt.record_and_transcribe.assert_called_once()

        finally:
            sys.stdout = sys.__stdout__

    def test_perform_recording_keyboard_interrupt(self) -> None:
        """Test handling of keyboard interrupt during recording."""
        mock_stt = MagicMock()
        mock_stt.record_and_transcribe.side_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            perform_recording(mock_stt, 1, 1)

    def test_perform_recording_exception(self) -> None:
        """Test handling of exception during recording."""
        mock_stt = MagicMock()
        mock_stt.record_and_transcribe.side_effect = Exception("Recording failed")

        # Should not raise exception, but print error
        captured_output = StringIO()
        sys.stderr = captured_output

        try:
            perform_recording(mock_stt, 1, 1)

            output = captured_output.getvalue()
            self.assertIn("Error during recording", output)

        finally:
            sys.stderr = sys.__stderr__


class TestMainFunction(unittest.TestCase):
    """Test cases for the main application function."""

    @patch("main.perform_recording")
    @patch("main.initialize_speech2text")
    @patch("main.parse_arguments")
    @patch("torch.cuda.is_available")
    def test_main_single_recording(
        self,
        mock_cuda: Mock,
        mock_parse: Mock,
        mock_init: Mock,
        mock_perform: Mock,
    ) -> None:
        """Test main function with single recording.

        Args:
            mock_cuda: Mocked CUDA availability check.
            mock_parse: Mocked argument parser.
            mock_init: Mocked Speech2Text initialization.
            mock_perform: Mocked recording execution.
        """
        # Setup mocks
        mock_cuda.return_value = False

        mock_args = MagicMock()
        mock_args.device = None
        mock_args.no_whisper_mic = False
        mock_args.verbose = False
        mock_args.recordings = 1
        mock_args.dtype = "float16"
        mock_parse.return_value = mock_args

        mock_stt = MagicMock()
        mock_init.return_value = mock_stt

        # Run main
        main.main()

        # Verify initialization
        mock_init.assert_called_once()

        # Verify recording was performed once
        mock_perform.assert_called_once_with(mock_stt, 1, 1)

    @patch("main.perform_recording")
    @patch("main.initialize_speech2text")
    @patch("main.parse_arguments")
    @patch("torch.cuda.is_available")
    @patch("time.sleep")
    def test_main_multiple_recordings(
        self,
        mock_sleep: Mock,
        mock_cuda: Mock,
        mock_parse: Mock,
        mock_init: Mock,
        mock_perform: Mock,
    ) -> None:
        """Test main function with multiple recordings.

        Args:
            mock_sleep: Mocked time.sleep function.
            mock_cuda: Mocked CUDA availability check.
            mock_parse: Mocked argument parser.
            mock_init: Mocked Speech2Text initialization.
            mock_perform: Mocked recording execution.
        """
        # Setup mocks
        mock_cuda.return_value = False

        mock_args = MagicMock()
        mock_args.device = None
        mock_args.no_whisper_mic = False
        mock_args.verbose = False
        mock_args.recordings = 3
        mock_args.dtype = "float16"
        mock_parse.return_value = mock_args

        mock_stt = MagicMock()
        mock_init.return_value = mock_stt

        # Run main
        main.main()

        # Verify recordings were performed 3 times
        self.assertEqual(mock_perform.call_count, 3)

        # Verify sleep was called between recordings (2 times for 3 recordings)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("main.initialize_speech2text")
    @patch("main.parse_arguments")
    @patch("torch.cuda.is_available")
    def test_main_initialization_failure(
        self,
        mock_cuda: Mock,
        mock_parse: Mock,
        mock_init: Mock,
    ) -> None:
        """Test main function with initialization failure.

        Args:
            mock_cuda: Mocked CUDA availability check.
            mock_parse: Mocked argument parser.
            mock_init: Mocked Speech2Text initialization that returns None.
        """
        # Setup mocks
        mock_cuda.return_value = False

        mock_args = MagicMock()
        mock_args.device = None
        mock_args.no_whisper_mic = False
        mock_args.verbose = False
        mock_args.recordings = 1
        mock_args.dtype = "float16"
        mock_parse.return_value = mock_args

        mock_init.return_value = None  # Initialization fails

        # Run main - should exit with code 1
        with self.assertRaises(SystemExit) as context:
            main.main()

        self.assertEqual(context.exception.code, 1)

    @patch("main.perform_recording")
    @patch("main.initialize_speech2text")
    @patch("main.parse_arguments")
    @patch("torch.cuda.is_available")
    def test_main_keyboard_interrupt(
        self,
        mock_cuda: Mock,
        mock_parse: Mock,
        mock_init: Mock,
        mock_perform: Mock,
    ) -> None:
        """Test main function handling of keyboard interrupt.

        Args:
            mock_cuda: Mocked CUDA availability check.
            mock_parse: Mocked argument parser.
            mock_init: Mocked Speech2Text initialization.
            mock_perform: Mocked recording that raises KeyboardInterrupt.
        """
        # Setup mocks
        mock_cuda.return_value = False

        mock_args = MagicMock()
        mock_args.device = None
        mock_args.no_whisper_mic = False
        mock_args.verbose = False
        mock_args.recordings = 1
        mock_args.dtype = "float16"
        mock_parse.return_value = mock_args

        mock_stt = MagicMock()
        mock_init.return_value = mock_stt

        mock_perform.side_effect = KeyboardInterrupt()

        # Run main - should exit with code 0
        with self.assertRaises(SystemExit) as context:
            main.main()

        self.assertEqual(context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
