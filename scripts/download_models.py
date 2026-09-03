#!/usr/bin/env python3
"""
Model Downloader for Sovereign AI Workbench
Downloads verified GGUF models directly from Hugging Face into the models/ folder.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

MODELS = {
    "coder-7b": {
        "filename": "qwen2.5-coder-7b-instruct-q3_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q3_k_m.gguf",
        "size_gb": 3.6,
        "description": "Primary recommended model for code generation, task planning, and deliverable creation.",
    },
    "general-7b": {
        "filename": "qwen2.5-7b-instruct-q3_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf",
        "size_gb": 3.6,
        "description": "General conversational and instruction model.",
    },
    "fallback-0.5b": {
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_gb": 0.46,
        "description": "Fast, lightweight emergency fallback model (<500MB, runs smoothly on any laptop CPU/GPU).",
    },
}


def print_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 35
        filled = int(bar_len * percent / 100.0)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({mb_down:6.1f} MB / {mb_total:6.1f} MB)")
        sys.stdout.flush()


def download_model(key: str, dest_dir: Path):
    if key not in MODELS:
        print(f"Unknown model: {key}. Available options: {list(MODELS.keys())}")
        return False

    info = MODELS[key]
    dest_path = dest_dir / info["filename"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 100 * 1024 * 1024:
        print(f"[OK] Model file already exists: {dest_path} ({dest_path.stat().st_size / (1024**3):.2f} GB)")
        return True

    print(f"\nDownloading {info['filename']} (~{info['size_gb']} GB)...")
    print(f"Description: {info['description']}")
    print(f"Source: {info['url']}\n")

    # Try using huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        print("Using huggingface_hub for optimized download with resume support...")
        downloaded_file = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        print(f"\n[OK] Successfully downloaded model to {downloaded_file}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Notice: huggingface_hub download encountered: {e}. Falling back to standard HTTPS stream...")

    # Direct urllib streaming fallback
    try:
        req = urllib.request.Request(
            info["url"],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SovereignAIWorkbench/1.0"}
        )
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
            total_size = int(resp.info().get("Content-Length", 0))
            block_size = 1024 * 1024  # 1MB
            downloaded = 0
            block_num = 0
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                block_num += 1
                print_progress(block_num, block_size, total_size)
        print(f"\n[OK] Successfully downloaded {info['filename']} to {dest_path}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {info['filename']}: {e}")
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for Sovereign AI Workbench")
    parser.add_argument(
        "--model",
        choices=["coder-7b", "general-7b", "fallback-0.5b", "all", "recommended"],
        default="recommended",
        help="Which model to download (default: recommended = coder-7b + fallback-0.5b)",
    )
    parser.add_argument(
        "--dest",
        default="models",
        help="Destination directory for model files (default: models)",
    )

    args = parser.parse_args()
    dest_dir = Path(args.dest)

    if args.model == "recommended":
        keys = ["fallback-0.5b", "coder-7b"]
    elif args.model == "all":
        keys = list(MODELS.keys())
    else:
        keys = [args.model]

    print("===================================================")
    print("   Sovereign AI Workbench - Model Downloader")
    print("===================================================")
    for k in keys:
        success = download_model(k, dest_dir)
        if not success:
            sys.exit(1)

    print("\nAll requested models are downloaded and ready in the models/ directory!")


if __name__ == "__main__":
    main()
