"""Main entry point for the speech2text package."""

import argparse
import sys
import time

import torch

from speech2text.config import AudioConfig, ModelConfig
from speech2text.speech2text import Speech2Text
from speech2text.utils.logging import get_logger

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Speech2Text Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (e.g., 'cuda', 'cpu')",
    )

    parser.add_argument(
        "--no-whisper-mic",
        action="store_true",
        help="Use local Hugging Face Whisper model instead of WhisperMic",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use",
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
        help="Number of recordings to perform",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Torch data type for model inference",
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_arguments()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model_config = ModelConfig(whisper_model=args.model_size)  # type: ignore
    audio_config = AudioConfig()

    try:
        with Speech2Text(
            device=device,
            torch_dtype=torch_dtype,
            use_whisper_mic=not args.no_whisper_mic,
            verbose=args.verbose,
            model_config=model_config,
            audio_config=audio_config,
        ) as stt:
            for i in range(1, args.recordings + 1):
                if args.verbose:
                    print(f"\n--- Recording {i}/{args.recordings} ---")
                text = stt.record_and_transcribe()
                print(f"Transcription: {text}")

                if i < args.recordings:
                    time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
