# Demo Datasets

Open, downloadable data for the four demo scenarios (agentic document task, coding task,
multimodal/vision task, RAG grounding). All data is stored locally in this folder — the
air-gapped workbench never needs external access once files are present.

Everything here can be re-fetched with:

```bash
./datasets/download_datasets.sh
```

> **License note:** check the dataset card of each source before commercial redistribution.
> Exact terms differ per corpus. US federal regulatory text (eCFR) is public domain.

---

## Layout & scenario map

| Scenario | Folder | Contents |
|---|---|---|
| Coding task (sandbox + verification) | `coding/` | HumanEval (164) + MBPP (~974) |
| Scanned inspection report → note (OCR/forms) | `scanned_docs/` | FUNSD (199 forms) + DocVQA subset + RVL-CDIP subset |
| P&ID / drawings (vision) | `pid/` | Digitize-PID sample (50 train P&IDs, YOLO labels, 32 symbol classes) |
| Handwritten notes (vision/OCR) | `handwriting/` | instructions only — IAM requires registration |
| RAG grounding (SOPs/regs) | `rag_docs/` | OSHA 29 CFR 1910.119/.134/.146 + EPA 40 CFR 68 (plain text) |

---

## 1. `coding/`

### HumanEval — `coding/humaneval/`
- 164 Python function-completion problems with unit tests. HF: `openai/openai_humaneval`.
- License: MIT (per OpenAI dataset card).

### MBPP — `coding/mbpp/`
- ~974 beginner programming problems with test cases. HF: `google-research-datasets/mbpp`
  (`sanitized/` = the standard benchmark split).
- License: CC-BY-4.0 (per Google MBPP dataset card).

---

## 2. `scanned_docs/`

### FUNSD — `scanned_docs/funsd/`
- 199 real scanned forms with entity-level labels (name, address, organisation, etc.) —
  the standard noisy-form / layout-parsing benchmark. HF: `nielsr/funsd`.
- License: free for research use (see dataset card).

### DocVQA sample — `scanned_docs/docvqa_1200/`
- 1,200-example subset (train/test parquet) of DocVQA — document visual question answering.
  HF: `nielsr/docvqa_1200_examples` (public subset of the full gated corpus).
- Full **DocVQA** (`docvqa/docvqa`) is gated on HF (agreement + login).

### RVL-CDIP sample — `scanned_docs/rvl_cdip_10_per_class/`
- 10 examples × 16 classes (letter, invoice, email, form, …) for scanned-document
  classification demos. HF: `nielsr/rvl_cdip_10_examples_per_class`.
- Full **RVL-CDIP** (~400K scanned business documents, tens of GB) is gated / huge.

---

## 3. `pid/digitize_pid/`

- 50 P&ID images (`images/train/`) + YOLO label files (`labels/train/`) covering all
  32 symbol classes of **Digitize-PID** (Paliwal et al., TCS Research).
- Downloaded from public mirror `hamzas/digitize-pid-yolo` on HF; the official
  `tcs-research/digitize-pid` repo is gated.
- Original Digitize-PID is under a custom (non-commercial research) license — check the
  mirror's README (included) for provenance.
- Demo tip: the original paper recommends testing symbol detection + OCR tag extraction;
  view any `images/train/*.jpg` against its `labels/train/*.txt` (YOLO `cls x y w h`).

---

## 4. `handwriting/`

**IAM Handwriting Database** (1,539 forms) is the standard handwriting benchmark but is
**registration-gated** (free for non-commercial research) — it cannot be downloaded
automatically. Visit the IAM site, register, and place the extracted forms here.

---

## 5. `rag_docs/`

US federal regulatory text (public domain) as dense, jargon-heavy grounding material.
Converted to plain text from eCFR XML (`_source/` holds the raw XML for provenance):

| File | What it is |
|---|---|
| `29cfr1910.119_osha.txt` | OSHA Process Safety Management of Highly Hazardous Chemicals |
| `29cfr1910.146_osha.txt` | OSHA Permit-Required Confined Spaces |
| `29cfr1910.134_osha.txt` | OSHA Respiratory Protection |
| `40cfr_part68_epa_rmp.txt` | EPA Risk Management Program (chemical accident prevention), 60 sections |

Stand-ins for real MRPL/refinery SOPs: same regulatory/jargon profile, no proprietary data.
To model "scanned inspection reports", pair the National Board **NB-5** form (public boiler
inspection form) with these — scan or print-at-low-DPI to add realistic noise.
Ingest into the workbench knowledge base via the app's `/ingest` endpoint (KB lives in
`data/knowledge_base/`).

---

## Not downloadable automatically (why, and how)

| Dataset | Blocker | How to obtain |
|---|---|---|
| IAM Handwriting | research registration required | register at the IAM site, download, drop into `handwriting/` |
| Roboflow Universe P&ID (~3,800 imgs) | account + API key | export via Roboflow CLI: `roboflow download ...` |
| Full RVL-CDIP | HF agreement gate + ~30 GB | accept terms on HF, then `huggingface-cli download` |
| Full DocVQA | HF agreement gate | accept terms on HF, then download |
| Official Digitize-PID repo | HF org gate | use the public mirror (done here) or request access |

Set `HF_TOKEN` in the environment if you later access gated HF repos.

---

## Totals

```text
du -sh datasets/*  (see per-folder sizes after download)
```

Retrieved: 2026-09-05. Sources reachable as of that date.
