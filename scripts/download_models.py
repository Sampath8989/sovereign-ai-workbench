#!/usr/bin/env python3
"""
Model Downloader for Sovereign AI Workbench
Downloads and installs verified GGUF models directly from Hugging Face or imports from local Ollama cache into models/.
Optimized for Pop!_OS / Linux and Windows with multi-threaded downloads, resume support, and 3B model variants.
"""

import os
import sys
import shutil
import json
import argparse
import subprocess
import urllib.request
from pathlib import Path

MODELS = {
    # 3B Models (Optimized for 4GB VRAM like RTX 2050 / Pop!_OS)
    "coder-3b-q4": {
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.1,
        "ollama_tag": "qwen2.5-coder:3b",
        "description": "Primary recommended 3B model for code generation, task planning, and deliverable creation.",
    },
    "coder-3b-q5": {
        "filename": "qwen2.5-coder-3b-instruct-q5_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q5_k_m.gguf",
        "size_gb": 2.3,
        "ollama_tag": None,
        "description": "High precision Q5_K_M variant of Qwen 2.5 Coder 3B for enhanced coding accuracy.",
    },
    "general-3b-q4": {
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "ollama_tag": "qwen2.5:3b",
        "description": "General conversational, reasoning, and instruction 3B model.",
    },
    "general-3b-q5": {
        "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf",
        "size_gb": 2.3,
        "ollama_tag": None,
        "description": "High precision Q5_K_M general instruction 3B model.",
    },
    "llama-3b-q4": {
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
        "ollama_tag": "llama3.2:3b",
        "description": "Meta Llama 3.2 3B Instruct - Lightweight, ultra-fast general reasoning model.",
    },
    "llama-3b-q5": {
        "filename": "llama-3.2-3b-instruct-q5_k_m.gguf",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "size_gb": 2.3,
        "ollama_tag": None,
        "description": "High precision Q5_K_M Meta Llama 3.2 3B Instruct.",
    },
    "fallback-0.5b": {
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_gb": 0.46,
        "ollama_tag": "qwen2.5:0.5b",
        "description": "Fast, lightweight emergency fallback model (<500MB, runs smoothly on any CPU/GPU).",
    },
    # 7B Legacy Models
    "coder-7b": {
        "filename": "qwen2.5-coder-7b-instruct-q3_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q3_k_m.gguf",
        "size_gb": 3.6,
        "ollama_tag": "qwen2.5-coder:7b",
        "description": "7B coding model (requires >= 4GB-8GB VRAM).",
    },
    "general-7b": {
        "filename": "qwen2.5-7b-instruct-q3_k_m.gguf",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf",
        "size_gb": 3.6,
        "ollama_tag": "qwen2.5:7b",
        "description": "7B general model (requires >= 4GB-8GB VRAM).",
    },
    "vl-7b": {
        "filename": "qwen2.5-vl-7b-instruct-q3_k_m.gguf",
        "repo_id": "bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF",
        "url": "https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen_Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf",
        "size_gb": 3.55,
        "ollama_tag": "qwen2.5vl:7b",
        "description": "Qwen 2.5 VL 7B vision-language model (Q3_K_M optimized for 4GB VRAM).",
    },
}

PRESETS = {
    "recommended": ["fallback-0.5b", "llama-3b-q4", "coder-3b-q4"],
    "3b-suite": ["coder-3b-q4", "llama-3b-q4", "general-3b-q4"],
    "all-3b": ["coder-3b-q4", "coder-3b-q5", "general-3b-q4", "general-3b-q5", "llama-3b-q4", "llama-3b-q5"],
    "all": list(MODELS.keys()),
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


def try_import_from_ollama(key: str, dest_path: Path) -> bool:
    """Check if model exists in local Ollama storage and copy it."""
    info = MODELS.get(key)
    if not info or not info.get("ollama_tag"):
        return False

    tag = info["ollama_tag"]
    parts = tag.split(":")
    model_name = parts[0]
    model_ver = parts[1] if len(parts) > 1 else "latest"

    # Search standard Ollama manifest locations
    search_dirs = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]

    for base in search_dirs:
        manifest_file = base / "manifests" / "registry.ollama.ai" / "library" / model_name / model_ver
        if manifest_file.exists():
            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                for layer in manifest.get("layers", []):
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        digest = layer.get("digest", "").replace("sha256:", "sha256-")
                        blob_path = base / "blobs" / digest
                        if blob_path.exists() and blob_path.stat().st_size > 100 * 1024 * 1024:
                            print(f"[+] Found {tag} in Ollama local cache ({blob_path.stat().st_size / (1024**3):.2f} GB)")
                            print(f"[*] Copying {blob_path.name} -> {dest_path.name} ...")
                            shutil.copyfile(str(blob_path), str(dest_path))
                            print(f"[OK] Successfully imported {dest_path.name} from local Ollama storage!")
                            return True
            except Exception as e:
                print(f"[-] Error reading Ollama manifest {manifest_file}: {e}")
    return False


def download_model(key: str, dest_dir: Path) -> bool:
    if key not in MODELS:
        print(f"Unknown model: {key}. Available options: {list(MODELS.keys())}")
        return False

    info = MODELS[key]
    dest_path = dest_dir / info["filename"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    expected_min_bytes = int(info["size_gb"] * 0.85 * (1024**3))
    if dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
        print(f"[OK] Model file already exists: {dest_path} ({dest_path.stat().st_size / (1024**3):.2f} GB)")
        return True

    # 1. Try local Ollama cache import first
    if try_import_from_ollama(key, dest_path):
        return True

    print(f"\nDownloading {info["filename"]} (~{info["size_gb"]} GB)...")
    print(f"Description: {info["description"]}")
    print(f"Source: {info["url"]}\n")

    # 2. Try aria2c for fast multi-connection download if available
    aria2c_bin = shutil.which("aria2c")
    if aria2c_bin:
        try:
            print("[*] Using aria2c with 16 parallel connections for fast download...")
            cmd = [
                aria2c_bin,
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-d", str(dest_dir),
                "-o", info["filename"],
                info["url"],
            ]
            res = subprocess.run(cmd)
            if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
                print(f"\n[OK] Successfully downloaded {info["filename"]} via aria2c")
                return True
        except Exception as e:
            print(f"[-] aria2c encountered: {e}. Falling back to huggingface_hub...")

    # 3. Try huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        print("[*] Using huggingface_hub for download with resume support...")
        downloaded_file = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        if dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
            print(f"\n[OK] Successfully downloaded model to {downloaded_file}")
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Notice: huggingface_hub encountered: {e}. Falling back to standard HTTPS stream...")

    # 4. Direct urllib streaming fallback
    try:
        req = urllib.request.Request(
            info["url"],
            headers={"User-Agent": "Mozilla/5.0 (Linux; Pop!_OS) SovereignAIWorkbench/1.0"}
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
        if dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
            print(f"\n[OK] Successfully downloaded {info["filename"]} to {dest_path}")
            return True
        else:
            print(f"\n[WARNING] Downloaded file is smaller than expected ({dest_path.stat().st_size} bytes)")
            return False
    except Exception as e:
        print(f"\n[ERROR] Failed to download {info["filename"]}: {e}")
        return False


def main():
    choices = list(MODELS.keys()) + list(PRESETS.keys())
    parser = argparse.ArgumentParser(description="Download GGUF models for Sovereign AI Workbench")
    parser.add_argument(
        "--model",
        choices=choices,
        default="recommended",
        help="Which model or preset to download (default: recommended = fallback-0.5b + llama-3b-q4 + coder-3b-q4)",
    )
    parser.add_argument(
        "--dest",
        default="models",
        help="Destination directory for model files (default: models)",
    )

    args = parser.parse_args()
    dest_dir = Path(args.dest)

    if args.model in PRESETS:
        keys = PRESETS[args.model]
    else:
        keys = [args.model]

    print("===================================================")
    print("   Sovereign AI Workbench - Model Downloader")
    print("   (Linux Pop!_OS & Multi-OS Optimized)")
    print("===================================================")
    for k in keys:
        success = download_model(k, dest_dir)
        if not success:
            print(f"[-] Warning: Failed or incomplete download for {k}")

    print("\nModel setup complete! Current models in directory:")
    if dest_dir.exists():
        for f in sorted(dest_dir.glob("*.gguf")):
            print(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
