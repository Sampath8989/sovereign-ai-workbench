#!/usr/bin/env bash
# ==============================================================================
# Download the open demo datasets into datasets/.
#   * All targets are public — no HF token required.
#   * Re-runnable: resumable, skips existing files where possible.
#   * Gated corpora (IAM, Roboflow, full RVL-CDIP/DocVQA/Digitize-PID) are NOT
#     fetched here; see datasets/README.md for how to obtain them.
#
# Usage:  ./datasets/download_datasets.sh
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p datasets/coding datasets/scanned_docs datasets/pid datasets/handwriting datasets/rag_docs/_source

# Python: prefer the project venv, fall back to system python3.
if [ -x "./venv/bin/python" ]; then PY="./venv/bin/python"; else PY="python3"; fi

# ----------------------------------------------------------------------------
# Hugging Face dataset snapshots (public).
# ----------------------------------------------------------------------------
hf_snapshot() {
    local repo="$1" local_dir="$2"
    echo ">> HF dataset: $repo -> $local_dir"
    HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" - "$repo" "$local_dir" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, local = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local)
print("   ok")
PYEOF
}

hf_snapshot "openai/openai_humaneval"        "datasets/coding/humaneval"
hf_snapshot "google-research-datasets/mbpp"  "datasets/coding/mbpp"
hf_snapshot "nielsr/funsd"                   "datasets/scanned_docs/funsd"
hf_snapshot "nielsr/docvqa_1200_examples"    "datasets/scanned_docs/docvqa_1200"
hf_snapshot "nielsr/rvl_cdip_10_examples_per_class" "datasets/scanned_docs/rvl_cdip_10_per_class"

# ----------------------------------------------------------------------------
# Digitize-PID (public mirror): fetch full mirror to staging, then assemble a
# deterministic demo subset (first 50 train + first 15 val by id).
# ----------------------------------------------------------------------------
PID_REPO="hamzas/digitize-pid-yolo"
PID_STAGE="datasets/pid/_full_pid"
PID_OUT="datasets/pid/digitize_pid"
echo ">> P&ID: staging full mirror ($PID_REPO)"
rm -rf "$PID_STAGE"
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" "$PID_STAGE" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="hamzas/digitize-pid-yolo", repo_type="dataset", local_dir=sys.argv[1])
PYEOF

echo ">> P&ID: assembling demo subset -> $PID_OUT"
rm -rf "$PID_OUT"
"$PY" "$PID_STAGE" "$PID_OUT" <<'PYEOF'
import os, shutil, sys
from pathlib import Path
stage, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p.relative_to(stage) for p in (stage / "DigitizePID_Dataset" / "images").rglob("*.jpg"))
def by_split(split, n):
    xs = sorted((f for f in files if f.parts[0] == split), key=lambda p: int(p.stem))
    return xs[:n]
chosen = by_split("train", 50) + by_split("val", 15)
for rel in chosen:
    split, name = rel.parts[0], rel.stem
    img_src = stage / "DigitizePID_Dataset" / "images" / split / f"{name}.jpg"
    lbl_src = stage / "DigitizePID_Dataset" / "labels" / split / f"{name}.txt"
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(img_src, out / "images" / split / f"{name}.jpg")
    if lbl_src.exists():
        shutil.copyfile(lbl_src, out / "labels" / split / f"{name}.txt")
shutil.copyfile(stage / "README.md", out / "README.md")
shutil.rmtree(stage)
print(f"   ok: {len(chosen)} images -> {out}")
PYEOF

# ----------------------------------------------------------------------------
# RAG grounding corpus: US federal regulatory text from eCFR (public domain).
# ----------------------------------------------------------------------------
echo ">> RAG docs: fetching eCFR XML (requires --compressed; eCFR refuses plain GETs)"
curl -sL --compressed --max-time 180 -A "Mozilla/5.0" \
    "https://www.ecfr.gov/api/versioner/v1/full/2024-01-01/title-40.xml?part=68" \
    -o datasets/rag_docs/_source/cfr40_part68_rmp.xml
curl -sL --compressed --max-time 300 -A "Mozilla/5.0" \
    "https://www.ecfr.gov/api/versioner/v1/full/2024-01-01/title-29.xml?part=1910" \
    -o datasets/rag_docs/_source/cfr29_part1910_osha.xml

echo ">> RAG docs: converting XML -> plain text"
"$PY" <<'PYEOF'
import xml.etree.ElementTree as ET
from pathlib import Path
SRC = Path("datasets/rag_docs/_source"); OUT = Path("datasets/rag_docs")
def walk_div8(el):
    if el.tag.endswith("DIV8"): yield el
    for c in el: yield from walk_div8(c)
def section_text(div8):
    lines = []
    for child in div8:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in ("HEAD",):
            lines.append("".join(child.itertext()).strip())
        elif tag in ("P", "FP", "HED", "DIV9", "DIV10", "DIV11", "DIV12", "TABLE"):
            t = " ".join("".join(child.itertext()).split())
            if t: lines.append(t)
    return "\n".join(x for x in lines if x)
root = ET.parse(SRC / "cfr40_part68_rmp.xml").getroot()
out = ["40 CFR Part 68 — Chemical Accident Prevention Provisions (EPA Risk Management Program)",
       "Source: eCFR (https://www.ecfr.gov), retrieved 2026-09-05. Public domain.", ""]
for s in walk_div8(root):
    t = section_text(s)
    if t: out += [t, ""]
(OUT / "40cfr_part68_epa_rmp.txt").write_text("\n".join(out))
root2 = ET.parse(SRC / "cfr29_part1910_osha.xml").getroot()
for sec in ("1910.119", "1910.146", "1910.134"):
    for s in walk_div8(root2):
        if s.get("N", "") == sec:
            head = f"29 CFR {sec} — OSHA (from 29 CFR Part 1910 XML)\nSource: eCFR, retrieved 2026-09-05. Public domain.\n\n"
            (OUT / f"29cfr{sec}_osha.txt").write_text(head + section_text(s))
            break
print("   ok: 40cfr_part68 + 29cfr1910.119/.134/.146")
PYEOF

echo
echo "Done. See datasets/README.md for sources, licenses, and gated items."
