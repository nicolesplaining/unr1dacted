#!/usr/bin/env python3
"""Plot type_1 vs type_2 counts per subject from typed DeepSeek results."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned.json")
OUTPUT_PNG = Path("results/type1_type2_by_subject.png")


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_counts(results: list[dict]) -> tuple[list[str], list[int], list[int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"type_1": 0, "type_2": 0})

    for item in results:
        label = item.get("type")
        if label not in {"type_1", "type_2"}:
            continue
        subject = str(item.get("subject", "unknown"))
        counts[subject][label] += 1

    subjects = sorted(counts.keys())
    type_1_counts = [counts[s]["type_1"] for s in subjects]
    type_2_counts = [counts[s]["type_2"] for s in subjects]
    return subjects, type_1_counts, type_2_counts


def plot(subjects: list[str], type_1_counts: list[int], type_2_counts: list[int], out_path: Path) -> None:
    if not subjects:
        raise ValueError("No type_1/type_2 records found in input JSON.")

    x = np.arange(len(subjects))
    width = 0.42

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width / 2, type_1_counts, width, label="type_1")
    ax.bar(x + width / 2, type_2_counts, width, label="type_2")

    ax.set_title("Type 1 vs Type 2 Counts by Subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    payload = load_payload(INPUT_PATH)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Expected 'results' list in input JSON.")

    subjects, type_1_counts, type_2_counts = compute_counts(results)
    plot(subjects, type_1_counts, type_2_counts, OUTPUT_PNG)
    print(f"Saved chart to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
