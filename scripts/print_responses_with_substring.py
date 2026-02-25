#!/usr/bin/env python3
"""
Print rows whose response field contains a target substring.
"""

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_FIELD = "leetspeak_response"
DEFAULT_NEEDLE = "4"


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print rows where a response field contains a substring."
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
        "--needle",
        type=str,
        default=DEFAULT_NEEDLE,
        help=f"Substring to search for (default: {DEFAULT_NEEDLE})",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive substring match.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Optional max number of matches to print (0 = no limit).",
    )
    args = parser.parse_args()

    if not args.needle:
        raise ValueError("--needle must be non-empty.")

    payload = load_payload(args.input)
    results = payload["results"]
    printed = 0

    needle = args.needle.lower() if args.ignore_case else args.needle

    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        response = row.get(args.field)
        if not isinstance(response, str) or not response:
            continue

        haystack = response.lower() if args.ignore_case else response
        if needle not in haystack:
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
