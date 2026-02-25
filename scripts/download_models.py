#!/usr/bin/env python3
"""
Model Download Script for ClearHelm

Downloads required GGUF models from Hugging Face to the local models directory.
Run: pip install -r requirements.txt && python scripts/download_models.py
"""

import os

# Enable hf_transfer for faster downloads (must be set before importing huggingface_hub)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from pathlib import Path
from huggingface_hub import hf_hub_download

# Base directory for models
MODELS_DIR = Path(__file__).parent.parent / "models"

# Models to download: (repo_id, filename, optional_subfolder)
MODELS = [
    {
        # https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
        "repo_id": "mradermacher/Qwen2.5-1.5B-GGUF",
        "filename": "Qwen2.5-1.5B.Q8_0.gguf",
        "description": "Qwen2.5 1.5B - Q8_0 quantization (1.7 GB)",
    },
    {
        # https://huggingface.co/mradermacher/Qwen2.5-7B-GGUF
        "repo_id": "mradermacher/Qwen2.5-7B-GGUF",
        "filename": "Qwen2.5-7B.Q8_0.gguf",
        "description": "Qwen2.5 7B - Q8_0 quantization (~7.7 GB)",
    },
    {
        # https://huggingface.co/mradermacher/Qwen2.5-7B-GGUF
        "repo_id": "mradermacher/Qwen2.5-7B-GGUF",
        "filename": "Qwen2.5-7B.Q4_K_M.gguf",
        "description": "Qwen2.5 7B - Q4_K_M quantization (~4.4 GB, faster inference)",
    },
    {
        # https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
        "repo_id": "unsloth/GLM-4.7-Flash-GGUF",
        "filename": "GLM-4.7-Flash-Q8_0.gguf",
        "description": "GLM-4 9B Flash - Q8_0 quantization (31.8 GB)",
    },
    {
        # https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q8_0.gguf",
        "description": "Qwen2.5 1.5B Instruct - Q8_0 quantization (1.65 GB)",
    },
]


def download_models():
    """Download all configured models to the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check hf_transfer availability
    try:
        import hf_transfer
        print("[OK] hf_transfer enabled (fast downloads)")
    except ImportError:
        print("[WARN] hf_transfer not installed - install with: pip install hf_transfer")

    print(f"Downloading models to: {MODELS_DIR.resolve()}")
    print("-" * 60)

    for model in MODELS:
        repo_id = model["repo_id"]
        filename = model["filename"]
        description = model.get("description", "")

        local_path = MODELS_DIR / filename

        if local_path.exists():
            print(f"[SKIP] {filename} already exists")
            continue

        print(f"\n[DOWNLOADING] {filename}")
        if description:
            print(f"  {description}")
        print(f"  From: {repo_id}")

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"[DONE] Saved to: {downloaded_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")

    print("\n" + "-" * 60)
    print("Download complete!")
    print(f"\nModels directory contents:")
    for f in MODELS_DIR.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_models()
