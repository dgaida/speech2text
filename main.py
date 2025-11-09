"""Speech2Text Example Application.

This script demonstrates the usage of the Speech2Text module with various
command-line options for recording and transcribing audio.

Examples:
    Basic usage with WhisperMic (default):
        $ python main.py

    Use local Hugging Face Whisper model:
        $ python main.py --no-whisper-mic

    Enable verbose output:
        $ python main.py --verbose

    Multiple recordings:
        $ python main.py --recordings 3

    Use CPU instead of GPU:
        $ python main.py --device cpu
"""

import argparse
import sys
import time
from typing import Optional

import torch

from speech2text import Speech2Text


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - device: Device to use for inference ('cuda' or 'cpu')
            - use_whisper_mic: Whether to use WhisperMic for recording
            - verbose: Enable verbose output
            - recordings: Number of recordings to perform
            - dtype: Torch data type for model inference
    """
    parser = argparse.ArgumentParser(
        description="Speech2Text Example Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use WhisperMic with default settings
  %(prog)s --no-whisper-mic         # Use local Whisper model
  %(prog)s --verbose                # Enable verbose output
  %(prog)s --recordings 3           # Perform 3 recordings
  %(prog)s --device cpu             # Force CPU usage
        """,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: auto-detect)",
    )

    parser.add_argument(
        "--no-whisper-mic",
        action="store_true",
        help="Use local Hugging Face Whisper model instead of WhisperMic",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--recordings",
        type=int,
        default=1,
        help="Number of recordings to perform (default: 1)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Torch data type for model inference (default: float16)",
    )

    return parser.parse_args()


def get_device(device_arg: Optional[str] = None) -> str:
    """Determine the device to use for inference.

    Args:
        device_arg: Optional device string from command-line arguments.
            If None, automatically detects CUDA availability.

    Returns:
        str: Device string ('cuda' or 'cpu').
    """
    if device_arg is not None:
        return device_arg

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for inference.")
        return "cuda"
    else:
        print("CUDA not available. Using CPU for inference.")
        return "cpu"


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype.

    Args:
        dtype_str: String representation of the dtype ('float16' or 'float32').

    Returns:
        torch.dtype: Corresponding torch data type.

    Raises:
        ValueError: If dtype_str is not recognized.
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Expected 'float16' or 'float32'.")

    return dtype_map[dtype_str]


def initialize_speech2text(
    device: str,
    torch_dtype: torch.dtype,
    use_whisper_mic: bool,
    verbose: bool,
) -> Optional[Speech2Text]:
    """Initialize the Speech2Text model.

    Args:
        device: Device to use for inference ('cuda' or 'cpu').
        torch_dtype: Torch data type for model inference.
        use_whisper_mic: Whether to use WhisperMic for recording.
        verbose: Enable verbose output.

    Returns:
        Optional[Speech2Text]: Initialized Speech2Text instance, or None if initialization fails.
    """
    try:
        print("\nInitializing Speech2Text model...")
        print(f"  Device: {device}")
        print(f"  Data type: {torch_dtype}")
        print(f"  Mode: {'WhisperMic' if use_whisper_mic else 'Local Whisper'}")

        stt = Speech2Text(
            device=device,
            torch_dtype=torch_dtype,
            use_whisper_mic=use_whisper_mic,
            verbose=verbose,
        )

        print("✓ Model initialized successfully.\n")
        return stt

    except Exception as e:
        print(f"✗ Error initializing Speech2Text: {e}", file=sys.stderr)
        return None


def perform_recording(stt: Speech2Text, recording_number: int, total_recordings: int) -> None:
    """Perform a single recording and transcription.

    Args:
        stt: Initialized Speech2Text instance.
        recording_number: Current recording number (1-indexed).
        total_recordings: Total number of recordings to perform.
    """
    print(f"\n{'='*60}")
    print(f"Recording {recording_number}/{total_recordings}")
    print(f"{'='*60}")

    try:
        start_time = time.time()
        text = stt.record_and_transcribe()
        elapsed_time = time.time() - start_time

        print(f"\n{'─'*60}")
        print("Transcribed text:")
        print(f"{'─'*60}")
        print(text)
        print(f"{'─'*60}")
        print(f"Processing time: {elapsed_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
        raise
    except Exception as e:
        print(f"\n✗ Error during recording: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point for the Speech2Text example application.

    Parses command-line arguments, initializes the Speech2Text model,
    and performs the specified number of recordings and transcriptions.
    """
    args = parse_arguments()

    # Determine device and data type
    device = get_device(args.device)
    torch_dtype = get_torch_dtype(args.dtype)
    use_whisper_mic = not args.no_whisper_mic

    # Initialize Speech2Text
    stt = initialize_speech2text(
        device=device,
        torch_dtype=torch_dtype,
        use_whisper_mic=use_whisper_mic,
        verbose=args.verbose,
    )

    if stt is None:
        sys.exit(1)

    # Perform recordings
    try:
        for i in range(1, args.recordings + 1):
            perform_recording(stt, i, args.recordings)

            # Wait between recordings if there are more to come
            if i < args.recordings:
                print("\nWaiting 2 seconds before next recording...")
                time.sleep(2)

        print(f"\n{'='*60}")
        print(f"✓ Completed {args.recordings} recording(s) successfully.")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
