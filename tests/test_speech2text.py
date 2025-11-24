"""Unit tests for the Speech2Text module.

This module contains comprehensive tests for the Speech2Text class,
including initialization, recording, transcription, and error handling.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from speech2text import Speech2Text


class TestSpeech2TextInitialization(unittest.TestCase):
    """Test cases for Speech2Text initialization."""

    @patch("speech2text.speech2text.WhisperMic")
    def test_init_with_whisper_mic(self, mock_whisper_mic: Mock) -> None:
        """Test initialization with WhisperMic enabled.

        Args:
            mock_whisper_mic: Mocked WhisperMic class.
        """
        mock_instance = MagicMock()
        mock_whisper_mic.return_value = mock_instance

        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

        # Verify WhisperMic was initialized
        mock_whisper_mic.assert_called_once_with(
            model="medium",
            english=False,
            pause=1,
            device="cpu",
        )

        # Verify attributes
        self.assertEqual(stt._whisper_mic, mock_instance)
        self.assertIsNone(stt._asr_model)
        self.assertFalse(stt.verbose())

    @patch("speech2text.speech2text.pipeline")
    def test_init_without_whisper_mic(self, mock_pipeline: Mock) -> None:
        """Test initialization with local Whisper model.

        Args:
            mock_pipeline: Mocked transformers pipeline function.
        """
        mock_model = MagicMock()
        mock_pipeline.return_value = mock_model

        stt = Speech2Text(
            device="cuda",
            torch_dtype=torch.float16,
            use_whisper_mic=False,
            verbose=True,
        )

        # Verify pipeline was called
        mock_pipeline.assert_called_once_with(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            torch_dtype=torch.float16,
            device="cuda",
        )

        # Verify attributes
        self.assertEqual(stt._asr_model, mock_model)
        self.assertIsNone(stt._whisper_mic)
        self.assertTrue(stt.verbose())

    @patch("speech2text.speech2text.WhisperMic")
    def test_init_whisper_mic_failure(self, mock_whisper_mic: Mock) -> None:
        """Test initialization fallback when WhisperMic fails.

        Args:
            mock_whisper_mic: Mocked WhisperMic class that raises an exception.
        """
        mock_whisper_mic.side_effect = AttributeError("CUDA error")

        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

        # Verify fallback to None
        self.assertIsNone(stt._whisper_mic)
        self.assertIsNone(stt._asr_model)


class TestSpeech2TextRecording(unittest.TestCase):
    """Test cases for Speech2Text recording functionality."""

    @patch("speech2text.speech2text.WhisperMic")
    def test_record_and_transcribe_whisper_mic(self, mock_whisper_mic: Mock) -> None:
        """Test recording with WhisperMic.

        Args:
            mock_whisper_mic: Mocked WhisperMic class.
        """
        mock_instance = MagicMock()
        mock_instance.listen.return_value = "Hello world"
        mock_whisper_mic.return_value = mock_instance

        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )

        result = stt.record_and_transcribe()

        self.assertEqual(result, "Hello world")
        mock_instance.listen.assert_called_once()

    @patch("speech2text.speech2text.pipeline")
    @patch("speech2text.speech2text.Speech2Text._record_audio_until_silence")
    @patch("speech2text.speech2text.write")
    @patch("speech2text.speech2text.os.remove")
    def test_record_and_transcribe_local_whisper(
        self,
        mock_remove: Mock,
        mock_write: Mock,
        mock_record: Mock,
        mock_pipeline: Mock,
    ) -> None:
        """Test recording with local Whisper model.

        Args:
            mock_remove: Mocked os.remove function.
            mock_write: Mocked scipy write function.
            mock_record: Mocked audio recording function.
            mock_pipeline: Mocked transformers pipeline function.
        """
        # Setup mocks
        mock_audio = np.random.rand(16000)
        mock_record.return_value = (mock_audio, 16000)

        mock_model = MagicMock()
        mock_model.return_value = {"text": "Test transcription"}
        mock_pipeline.return_value = mock_model

        # Initialize and test
        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=False,
            verbose=False,
        )

        result = stt.record_and_transcribe()

        # Verify result
        self.assertEqual(result, "Test transcription")

        # Verify audio was recorded
        mock_record.assert_called_once()

        # Verify audio file was written
        self.assertEqual(mock_write.call_count, 1)

        # Verify model was called with generate_kwargs
        self.assertEqual(mock_model.call_count, 1)
        call_args = mock_model.call_args
        self.assertIn("generate_kwargs", call_args.kwargs)
        self.assertEqual(call_args.kwargs["generate_kwargs"], {"task": "translate"})

        # Verify temp file was removed
        self.assertEqual(mock_remove.call_count, 1)


class TestRecordAudioUntilSilence(unittest.TestCase):
    """Test cases for audio recording with silence detection."""

    @patch("speech2text.speech2text.sd.InputStream")
    @patch("speech2text.speech2text.time.time")
    def test_record_audio_until_silence(self, mock_time: Mock, mock_stream: Mock) -> None:
        """Test audio recording stops after detecting silence.

        Args:
            mock_time: Mocked time.time function.
            mock_stream: Mocked sounddevice InputStream.
        """
        # Create mock audio chunks (loud then quiet)
        loud_chunk = np.random.rand(1600, 1) * 0.1
        quiet_chunk = np.random.rand(1600, 1) * 0.0001

        # Mock stream to return loud chunks then quiet chunks
        mock_stream_instance = MagicMock()
        mock_stream_instance.read.side_effect = [
            (loud_chunk, None),  # Loud chunk 1
            (loud_chunk, None),  # Loud chunk 2 (resets silence)
            (quiet_chunk, None),  # Quiet chunk 1 (starts silence timer)
            (quiet_chunk, None),  # Quiet chunk 2 (continues silence)
            (quiet_chunk, None),  # Quiet chunk 3 (continues silence)
            (quiet_chunk, None),  # Quiet chunk 4 (triggers stop at 3.1s)
        ]

        mock_stream.return_value.__enter__.return_value = mock_stream_instance

        # Mock time to simulate silence duration
        # Each quiet chunk needs a time check: initial set + check after each quiet chunk
        time_values = [
            0.0,  # First loud chunk
            0.5,  # Second loud chunk
            1.0,  # First quiet chunk - silence_start set to 1.0
            2.0,  # Check after first quiet: 2.0 - 1.0 = 1.0 < 3.0 (continue)
            2.5,  # Second quiet chunk
            2.8,  # Check after second quiet: 2.8 - 1.0 = 1.8 < 3.0 (continue)
            3.0,  # Third quiet chunk
            3.5,  # Check after third quiet: 3.5 - 1.0 = 2.5 < 3.0 (continue)
            3.8,  # Fourth quiet chunk
            4.2,  # Check after fourth quiet: 4.2 - 1.0 = 3.2 >= 3.0 (STOP)
        ]

        mock_time.side_effect = time_values

        # Record audio
        audio_data, sample_rate = Speech2Text._record_audio_until_silence(
            silence_threshold=0.001,
            silence_duration=3.0,
        )

        # Verify results
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_data, np.ndarray)
        self.assertGreater(len(audio_data), 0)

        # Verify the stream was read the expected number of times (6 chunks total)
        self.assertEqual(mock_stream_instance.read.call_count, 6)

    @patch("speech2text.speech2text.sd.InputStream")
    def test_record_audio_no_silence(self, mock_stream: Mock) -> None:
        """Test audio recording with continuous loud audio.

        Args:
            mock_stream: Mocked sounddevice InputStream.
        """
        # Create only loud chunks
        loud_chunk = np.random.rand(1600, 1) * 0.1

        mock_stream_instance = MagicMock()
        # Limit iterations to prevent infinite loop in test
        mock_stream_instance.read.side_effect = [(loud_chunk, None)] * 100

        mock_stream.return_value.__enter__.return_value = mock_stream_instance

        # This would normally run indefinitely, so we limit it with a timeout
        with self.assertRaises(StopIteration):
            Speech2Text._record_audio_until_silence(
                silence_threshold=0.001,
                silence_duration=3.0,
            )


class TestSpeech2TextVerbose(unittest.TestCase):
    """Test cases for verbose mode."""

    @patch("speech2text.speech2text.WhisperMic")
    def test_verbose_property(self, mock_whisper_mic: Mock) -> None:
        """Test verbose property getter.

        Args:
            mock_whisper_mic: Mocked WhisperMic class.
        """
        mock_whisper_mic.return_value = MagicMock()

        # Test verbose=False
        stt_quiet = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=False,
        )
        self.assertFalse(stt_quiet.verbose())

        # Test verbose=True
        stt_verbose = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=True,
            verbose=True,
        )
        self.assertTrue(stt_verbose.verbose())


class TestSpeech2TextIntegration(unittest.TestCase):
    """Integration tests for Speech2Text (require actual dependencies)."""

    def test_torch_dtype_compatibility(self) -> None:
        """Test that different torch dtypes can be passed."""
        dtypes = [torch.float16, torch.float32, torch.bfloat16]

        for dtype in dtypes:
            # This should not raise an error
            try:
                with patch("speech2text.speech2text.pipeline") as mock_pipeline:
                    mock_pipeline.return_value = MagicMock()
                    Speech2Text(
                        device="cpu",
                        torch_dtype=dtype,
                        use_whisper_mic=False,
                        verbose=False,
                    )
            except Exception as e:
                self.fail(f"Failed with dtype {dtype}: {e}")


class TestSpeech2TextEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @patch("speech2text.speech2text.pipeline")
    @patch("speech2text.speech2text.Speech2Text._record_audio_until_silence")
    def test_empty_audio_recording(self, mock_record: Mock, mock_pipeline: Mock) -> None:
        """Test handling of empty audio recording.

        Args:
            mock_record: Mocked audio recording function.
            mock_pipeline: Mocked transformers pipeline function.
        """
        # Return empty audio
        mock_record.return_value = (np.array([]), 16000)

        mock_model = MagicMock()
        mock_model.return_value = {"text": ""}
        mock_pipeline.return_value = mock_model

        stt = Speech2Text(
            device="cpu",
            torch_dtype=torch.float32,
            use_whisper_mic=False,
            verbose=False,
        )

        # Should handle empty audio gracefully
        with patch("speech2text.speech2text.write"):
            with patch("speech2text.speech2text.os.remove"):
                result = stt.record_and_transcribe()
                self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
