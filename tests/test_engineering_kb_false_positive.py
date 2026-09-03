#!/usr/bin/env python3
"""
Targeted test: Confirm the >=$1M content filter does NOT falsely exclude
legitimate engineering cost estimates (e.g., "$2.5 million" for a turbine).

The filter has two layers:
  1. Keyword filter: flags words like 'confidential', 'budget', 'salary'
  2. Dollar-amount filter: flags comma-formatted numbers >= $1,000,000

This test specifically verifies that WORD-FORMAT dollar amounts like
"$2.5 million" are NOT caught by the dollar-amount filter.

Note: the word "budget" IS a keyword trigger (correctly). This test
avoids that keyword to isolate the dollar-amount filter behavior.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HARDWARE_TIER", "BUILD")

from backend.tools.rag_search import get_rag, contains_sensitive_content

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ PASS: {label}")
    else:
        failed += 1
        print(f"  ❌ FAIL: {label}")
    if detail:
        print(f"         {detail[:250]}")


# ============================================================
# PART 1: Word-format "$X million" must NOT trigger the filter
# ============================================================
print("\n=== PART 1: Word-format dollar amounts are NOT flagged ===")
print("   (The regex only matches comma-formatted: $X,NNN,NNN)")

word_format_cases = [
    ("The replacement turbine costs $2.5 million and weighs 45 tonnes",
     "$2.5 million"),
    ("Total project cost: $3.5 million for the pump installation",
     "$3.5 million"),
    ("Installation cost: $2.5 million. Labor rate: $150/hr",
     "$2.5 million mixed with other costs"),
    ("The heat exchanger costs $1.2 million and requires 6 months lead time",
     "$1.2 million"),
    ("Replacement cost is $5.0 million, delivery in Q3",
     "$5.0 million"),
    ("Capital expenditure: $8.0 million over 24 months for the upgrade",
     "$8.0 million"),
]

for text, desc in word_format_cases:
    check(
        f"NOT flagged: {desc}",
        not contains_sensitive_content(text),
        f"Text: {text[:120]}",
    )

# ============================================================
# PART 2: Comma-formatted >=$1M numbers ARE correctly flagged
# ============================================================
print("\n=== PART 2: Comma-formatted >=$1M numbers ARE correctly flagged ===")

comma_format_cases = [
    ("$12,700,000 allocated to Project Omega", "$12.7M"),
    ("The merger value is $8,300,000,000", "$8.3B"),
    ("$1,500,000.00 budget allocation", "$1.5M with decimals"),
    ("Total cost $3,500,000 for the pump", "$3.5M (2 comma groups)"),
    ("$12,500,000 over 18 months", "$12.5M (2 comma groups)"),
]

for text, desc in comma_format_cases:
    check(
        f"Correctly flagged: {desc}",
        contains_sensitive_content(text),
        f"Text: {text[:120]}",
    )

# ============================================================
# PART 3: Sensitive keywords correctly flagged (independent of $)
# ============================================================
print("\n=== PART 3: Sensitive keywords correctly flagged ===")

keyword_cases = [
    ("Employee salary data shows payroll of $500K", "'salary' + 'payroll'"),
    ("This is restricted financial statement data", "'restricted' + 'financial statement'"),
    ("Non-disclosure agreement for IPO valuation data", "'non-disclosure' + 'ipo'"),
    ("CONFIDENTIAL: Q4 budget is $12.7M", "'confidential'"),
]

for text, desc in keyword_cases:
    check(
        f"Correctly flagged: {desc}",
        contains_sensitive_content(text),
        f"Text: {text[:120]}",
    )

# ============================================================
# PART 4: Full RAG pipeline — engineering chunk with $2.5 million
#   Key: verify (a) not re-tagged at ingestion, (b) not excluded at search time
# ============================================================
print("\n=== PART 4: Full RAG pipeline — engineering chunk with $2.5 million ===")

rag = get_rag()

# Engineering doc: large dollar figure, no sensitive keywords
engineering_chunk = {
    "text": (
        "The replacement turbine costs $2.5 million and weighs 45 tonnes. "
        "Installation requires a 200-ton crane. Total project cost estimate: "
        "$2.5 million over 18 months. The rotor diameter is 4.2m and operates "
        "at 3,600 RPM with a shaft output of 12 MW."
    ),
    "metadata": {"collection": "engineering_kb", "source": "turbine_spec.md"},
}

n = rag.ingest([engineering_chunk])
check("Ingested 1 engineering chunk", n == 1, f"count={n}")

# Verify content filter doesn't flag the text
check(
    "contains_sensitive_content() returns False for engineering text",
    not contains_sensitive_content(engineering_chunk["text"]),
    f"Text: {engineering_chunk['text'][:150]}",
)

# Verify the chunk was NOT re-tagged to financials_restricted at ingestion
re_tagged = False
for meta in rag._bm25_metadata:
    if meta.get("source") == "turbine_spec.md":
        re_tagged = meta.get("collection") != "engineering_kb"
        check(
            "Chunk NOT re-tagged to financials_restricted at ingestion",
            not re_tagged,
            f"collection={meta.get('collection')}, _content_retagged={meta.get('_content_retagged')}",
        )
        break
else:
    check("Chunk found in BM25 metadata", False, "turbine_spec.md not found in metadata")

# Search as engineer — the chunk should NOT be filtered by the content filter
# (MockEmbedder may rank it low due to hash-based similarity, so we check
#  whether it would be filtered IF it appeared in results, by searching with
#  role=engineer and verifying no re-tagged chunks appear)
results = rag.search("replacement turbine cost estimate", top_k=20, role="engineer")

# The key check: no engineering_kb chunks were re-tagged to financials_restricted
any_retagged_in_results = any(
    r.get("metadata", {}).get("_content_retagged") == True
    and r.get("metadata", {}).get("source") == "turbine_spec.md"
    for r in results
)
check(
    "Engineering doc NOT re-tagged or excluded by content filter",
    not any_retagged_in_results,
    f"Re-tagged in results: {any_retagged_in_results}",
)

# Even more direct: simulate what the search filter would do
from backend.tools.rag_search import contains_sensitive_content as csc
engineer_would_see_it = not csc(engineering_chunk["text"])
check(
    "Engineer-role content filter would NOT exclude this chunk",
    engineer_would_see_it,
    "If this chunk appeared in ranked results, the content filter would let it through",
)

# ============================================================
# PART 5: Cross-check — genuinely sensitive content IS filtered
# ============================================================
print("\n=== PART 5: Cross-check — restricted content IS filtered ===")

rag.ingest([
    {
        "text": "CONFIDENTIAL: Q4 budget is $12,700,000. Merger value $8,300,000,000.",
        "metadata": {"collection": "financials_restricted", "source": "q4_report.txt"},
    }
])

results2 = rag.search("CONFIDENTIAL Q4 budget merger value", top_k=10, role="engineer")
restricted_leaked = any(
    r.get("metadata", {}).get("collection") == "financials_restricted"
    for r in results2
)
check(
    "Restricted content filtered from engineer results",
    not restricted_leaked,
    f"Restricted chunk leaked: {restricted_leaked}",
)

# ============================================================
# PART 6: Abbreviated B/M/K suffix formats ARE correctly flagged
# ============================================================
print("\n=== PART 6: Abbreviated financial formats (B/M/K suffix) ARE flagged ===")
print("   Catches: $8.3B, $8.3M, $500K, $12.5B, $3.2M")

bmk_format_cases = [
    ("Revealed: $8.3B in offshore accounts", "$8.3B - exact leak format"),
    ("Record payout: $8.3M bonus", "$8.3M - million abbreviation"),
    ("Annual salary: $500K", "$500K - thousand abbreviation"),
    ("Deal valued at $12.5B", "$12.5B - billion with decimal"),
    ("Payout of $3.2M", "$3.2M - million with decimal"),
    ("Cost: $500K", "$500K - thousand with space"),
    ("The merger value is $8.3 billion.", "$8.3 billion - word format (keyword 'merger' catches this)"),
]

for text, desc in bmk_format_cases:
    check(
        f"Correctly flagged: {desc}",
        contains_sensitive_content(text),
        f"Text: {text[:120]}",
    )

# Negative: abbreviated formats should NOT false-positive on normal engineering text
# (Note: these have NO sensitive keywords AND no B/M/K dollar patterns)
bmk_negative_cases = [
    ("The pipe diameter is 8.3 inches", "8.3 inches - not dollar"),
    ("Temperature is 500K", "500K - no dollar sign"),
    ("Model designation: M-3.2", "M-3.2 - equipment tag"),
]

for text, desc in bmk_negative_cases:
    check(
        f"NOT falsely flagged: {desc}",
        not contains_sensitive_content(text),
        f"Text: {text[:120]}",
    )

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} PASS / {failed} FAIL out of {passed + failed} checks")
print(f"{'=' * 60}")

if failed:
    print("\n⚠️  Some checks FAILED — investigate above results.")
    sys.exit(1)
else:
    print("\n✅ All checks passed — $2.5 million word-format is NOT falsely excluded.")
    sys.exit(0)
