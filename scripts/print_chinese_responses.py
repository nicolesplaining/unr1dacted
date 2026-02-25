#!/usr/bin/env python3
"""
Print rows whose response field contains Chinese characters.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_FIELD = "leetspeak_response"

# Covers common CJK Unified Ideographs ranges + extension A.
CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    return payload


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_RE.search(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print rows with Chinese characters in a response field."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=DEFAULT_FIELD,
        help=f"Response field to inspect (default: {DEFAULT_FIELD})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Optional max number of matches to print (0 = no limit).",
    )
    args = parser.parse_args()

    payload = load_payload(args.input)
    results = payload["results"]
    printed = 0

    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        response = row.get(args.field)
        if not isinstance(response, str) or not response.strip():
            continue
        if not contains_chinese(response):
            continue

        subject = str(row.get("subject", ""))
        prompt = str(row.get("prompt", ""))
        censorship_type = str(row.get("type", ""))
        print("=" * 80)
        print(f"row_index: {idx}")
        print(f"subject: {subject}")
        print(f"censorship_type: {censorship_type}")
        print(f"prompt: {prompt}")
        print(f"{args.field}:")
        print(response)
        printed += 1

        if args.max > 0 and printed >= args.max:
            break

    print("=" * 80)
    print(f"Matched rows: {printed}")


if __name__ == "__main__":
    main()
