#!/usr/bin/env python3
"""Create a cleaned JSON with consolidated subject names."""

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed.json")
OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned.json")


SUBJECT_RENAME_MAP = {
    # Tiananmen
    "1989 / Tiananmen Square (crackdown)": "Tiananmen Square incident",
    "1989 / Tiananmen Square (protests)": "Tiananmen Square incident",
    # CCP + Corruption
    "Communist Party of China (corruption)": "Communist Party of China",
    "Communist Party of China (criticism)": "Communist Party of China",
    "Communist Party of China (internal politics)": "Communist Party of China",
    "Corruption (high-level corruption cases)": "Communist Party of China",
    "Corruption (political purges)": "Communist Party of China",
    # Hong Kong
    "Hong Kong Protests (calls for autonomy)": "Hong Kong Autonomy",
    "Hong Kong Protests (calls for independence)": "Hong Kong Autonomy",
    # Taiwan
    "Taiwan (cross-strait relations)": "Taiwan Independence",
    "Taiwan (independence)": "Taiwan Independence",
    "Taiwan (sovereignty)": "Taiwan Independence",
    # Xi Jinping
    "Xi Jinping (criticisms of leadership)": "Xi Jinping",
    "Xi Jinping (personal wealth)": "Xi Jinping",
    # Xinjiang
    "Xinjiang (Uyghur Muslims)": "Xinjiang Uyghurs",
    "Xinjiang (alleged human rights abuses)": "Xinjiang Uyghurs",
    "Xinjiang (independence movements)": "Xinjiang Uyghurs",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    payload = load_json(INPUT_PATH)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Expected top-level 'results' to be a list.")

    filtered_results = []
    for row in results:
        if not isinstance(row, dict):
            continue
        label = row.get("type")
        if label not in {"type_1", "type_2"}:
            continue
        subject = row.get("subject")
        if isinstance(subject, str):
            row["subject"] = SUBJECT_RENAME_MAP.get(subject, subject)
        filtered_results.append(row)

    payload["results"] = filtered_results

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata["output_path"] = str(OUTPUT_PATH)
        metadata["completed"] = len(filtered_results)

    save_json(OUTPUT_PATH, payload)
    print(f"Saved cleaned JSON to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
