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
    type_1_pct: list[int] = []
    type_2_pct: list[int] = []
    for subject in subjects:
        c1 = counts[subject]["type_1"]
        c2 = counts[subject]["type_2"]
        total = c1 + c2
        if total == 0:
            type_1_pct.append(0)
            type_2_pct.append(0)
        else:
            # Round type_1 to nearest integer and set type_2 so stacks sum to 100.
            c1_pct = round((c1 / total) * 100.0)
            c2_pct = 100 - c1_pct
            type_1_pct.append(c1_pct)
            type_2_pct.append(c2_pct)
    return subjects, type_1_pct, type_2_pct


def plot(subjects: list[str], type_1_pct: list[float], type_2_pct: list[float], out_path: Path) -> None:
    if not subjects:
        raise ValueError("No type_1/type_2 records found in input JSON.")

    x = np.arange(len(subjects))
    width = 0.6

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(x, type_1_pct, width, label="type_1")
    ax.bar(x, type_2_pct, width, bottom=type_1_pct, label="type_2")

    ax.set_title("Type 1 vs Type 2 Censorship Count by Topic", fontsize=22)
    ax.set_xlabel("Topic", fontsize=16)
    ax.set_ylabel("Percent", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=14)
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

    subjects, type_1_pct, type_2_pct = compute_counts(results)
    plot(subjects, type_1_pct, type_2_pct, OUTPUT_PNG)
    print(f"Saved chart to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
