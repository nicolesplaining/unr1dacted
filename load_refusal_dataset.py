#!/usr/bin/env python3
"""Load and explore the LLM refusal evaluation dataset from Hugging Face.

Login using e.g. `huggingface-cli login` to access this dataset.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from datasets import DatasetDict, load_dataset


DATASET_ID = "MultiverseComputingCAI/llm-refusal-evaluation"
SUBJECT_CANDIDATES = ("subject", "category", "topic", "domain", "type", "label")
PROMPT_CANDIDATES = ("prompt", "question", "input", "instruction", "query", "text")
TARGET_SUBJECT = "ccp_sensitive"
OUTPUT_JSON_PATH = Path("data/ccp_sensitive_prompts.json")
TARGET_TOPICS = {
    "1989 / Tiananmen Square (crackdown)",
    "1989 / Tiananmen Square (protests)",
    "Communist Party of China (corruption)",
    "Communist Party of China (criticism)",
    "Communist Party of China (internal politics)",
    "Corruption (high-level corruption cases)",
    "Corruption (political purges)",
    "Hong Kong Protests (2019-2020 protests)",
    "Hong Kong Protests (calls for autonomy)",
    "Hong Kong Protests (calls for independence)",
    "Taiwan (cross-strait relations)",
    "Taiwan (independence)",
    "Taiwan (sovereignty)",
    "The Cultural Revolution (1966-1976)",
    "Xi Jinping (criticisms of leadership)",
    "Xi Jinping (personal wealth)",
    "Xi Jinping (\"Winnie the Pooh\" references)",
    "Xinjiang (Uyghur Muslims)",
    "Xinjiang (alleged human rights abuses)",
    "Xinjiang (independence movements)",
}


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


def detect_prompt_column(
    ds: DatasetDict, candidates: Sequence[str] = PROMPT_CANDIDATES
) -> Optional[str]:
    """Return the first matching prompt-like column name."""
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


def keep_only_topics(ds: DatasetDict, allowed_topics: set[str]) -> DatasetDict:
    """Keep only rows with subject values in the allowed list."""
    subject_col = detect_subject_column(ds)
    if subject_col is None:
        raise ValueError(
            "Could not find a subject-like column. "
            "Update SUBJECT_CANDIDATES to match your dataset schema."
        )

    normalized_allowed = {topic.strip().lower() for topic in allowed_topics}

    filtered = ds.filter(
        lambda example: str(example[subject_col]).strip().lower() in normalized_allowed
    )
    return filtered


def save_subject_and_prompts_json(ds: DatasetDict, output_path: Path) -> None:
    """Save filtered rows as JSON records with only subject and prompt."""
    subject_col = detect_subject_column(ds)
    prompt_col = detect_prompt_column(ds)
    if subject_col is None or prompt_col is None:
        first_split = next(iter(ds.values()))
        raise ValueError(
            "Could not detect subject/prompt columns. "
            f"Available columns: {first_split.column_names}"
        )

    records: list[dict[str, str]] = []
    for _, split_ds in ds.items():
        for row in split_ds:
            records.append(
                {
                    "subject": str(row[subject_col]),
                    "prompt": str(row[prompt_col]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} records to '{output_path}'")
    print()


def main() -> None:
    ds = load_refusal_dataset()
    ds = keep_only_split(ds, TARGET_SUBJECT)
    ds = keep_only_topics(ds, TARGET_TOPICS)
    print(f"Kept only split='{TARGET_SUBJECT}'")
    print()
    subject_col = detect_subject_column(ds)
    if subject_col:
        present_topics = {str(v) for v in ds[TARGET_SUBJECT][subject_col]}
        missing_topics = sorted(topic for topic in TARGET_TOPICS if topic not in present_topics)
        if missing_topics:
            print("Requested topics not present in filtered data:")
            for topic in missing_topics:
                print(f"  - {topic}")
            print()
    save_subject_and_prompts_json(ds, OUTPUT_JSON_PATH)
    explore_dataset(ds)


if __name__ == "__main__":
    main()
