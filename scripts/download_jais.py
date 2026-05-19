#!/usr/bin/env python3
"""Download Jais-13B-chat 4-bit from HuggingFace for local use."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "mlconvexai/jais-13b-chat_bitsandbytes_4bit"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "mlconvexai" / "jais-13b-chat_bitsandbytes_4bit"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {MODEL_DIR}...")
    print("This is a ~7-8GB download (4-bit quantized).")
    print("This model is NOT gated, no license acceptance needed.")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        print(f"Failed to download model: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Model downloaded to {MODEL_DIR}")
    print("Update config.yaml to use the local path:")
    print(f"  model_name_or_path: {MODEL_DIR}")
    print("  local_files_only: true")


if __name__ == "__main__":
    main()
