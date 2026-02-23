#!/usr/bin/env python3
"""Load and explore the LLM refusal evaluation dataset from Hugging Face.

Login using e.g. `huggingface-cli login` to access this dataset.
"""

from collections import Counter
from typing import Optional, Sequence

from datasets import DatasetDict, load_dataset


DATASET_ID = "MultiverseComputingCAI/llm-refusal-evaluation"
SUBJECT_CANDIDATES = ("subject", "category", "topic", "domain", "type", "label")
TARGET_SUBJECT = "ccp_sensitive"


def load_refusal_dataset(dataset_id: str = DATASET_ID) -> DatasetDict:
    """Load the target dataset from Hugging Face."""
    return load_dataset(dataset_id)


def detect_subject_column(
    ds: DatasetDict, candidates: Sequence[str] = SUBJECT_CANDIDATES
) -> Optional[str]:
    """Return the first matching subject-like column name."""
    for _, split_ds in ds.items():
        for col in candidates:
            if col in split_ds.column_names:
                return col
    return None


def explore_dataset(ds: DatasetDict) -> None:
    """Print split information and counts by subject-like column."""
    print("Dataset splits:", list(ds.keys()))
    print()

    subject_col = detect_subject_column(ds)
    if subject_col is None:
        first_split = next(iter(ds.values()))
        print("Columns:", first_split.column_names)
        print("(No standard 'subject' column found; adjust SUBJECT_CANDIDATES if needed.)")
        return

    print(f"Subject column: '{subject_col}'")
    print()

    for split_name, split_ds in ds.items():
        print(f"--- {split_name} (n={len(split_ds)}) ---")
        counts = Counter(split_ds[subject_col])
        for subject, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
            print(f"  {subject}: {count}")
        print()


def keep_only_split(ds: DatasetDict, split_name: str) -> DatasetDict:
    """Keep only one split from the dataset dict."""
    if split_name not in ds:
        available = ", ".join(ds.keys())
        raise ValueError(f"Split '{split_name}' not found. Available splits: {available}")
    return DatasetDict({split_name: ds[split_name]})


def main() -> None:
    ds = load_refusal_dataset()
    ds = keep_only_split(ds, TARGET_SUBJECT)
    print(f"Kept only split='{TARGET_SUBJECT}'")
    print()
    explore_dataset(ds)


if __name__ == "__main__":
    main()
