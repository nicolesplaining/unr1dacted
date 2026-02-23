#!/usr/bin/env python3
"""
Run DeepSeek-R1 (API) on prompts from the cleaned results JSON, after prompt edits:
1) remove trailing period
2) append " but in leetspeak"

Writes output incrementally after each response.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    # Allow running without python-dotenv if env vars are already exported.
    def load_dotenv() -> bool:
        return False


DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned.json")
DEFAULT_OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")


def transform_prompt(prompt: str) -> str:
    """Remove final period if present, then append required suffix."""
    trimmed = prompt.rstrip()
    if trimmed.endswith("."):
        trimmed = trimmed[:-1]
    return f"{trimmed} but in leetspeak"


def load_input_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"Expected 'results' list in '{path}'")
    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        payload["metadata"] = {}
    return payload


def load_existing_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metadata": {}, "results": []}

    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Expected object in existing results file: '{path}'")
    if "results" not in existing or not isinstance(existing["results"], list):
        raise ValueError(f"Existing file '{path}' is missing a valid 'results' list.")
    if "metadata" not in existing or not isinstance(existing["metadata"], dict):
        existing["metadata"] = {}
    return existing


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def prepare_results_rows(payload: dict[str, Any]) -> None:
    """Ensure each row has leetspeak fields while preserving all original fields."""
    for row in payload["results"]:
        if not isinstance(row, dict):
            continue
        original_prompt = str(row.get("prompt", "")).strip()
        row.setdefault("leetspeak_prompt", transform_prompt(original_prompt))
        row.setdefault("leetspeak_response", None)
        row.setdefault("leetspeak_error", None)
        row.setdefault("leetspeak_model", None)
        row.setdefault("leetspeak_created_at", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch DeepSeek-R1 API runner with leetspeak prompt edits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input cleaned JSON path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output results JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        default="deepseek-reasoner",
        help="DeepSeek model id (default: deepseek-reasoner)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for completion.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt for all calls.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Export it or add it to a .env file."
        )

    source_payload = load_input_payload(args.input)

    if args.output.exists():
        payload = load_existing_results(args.output)
    else:
        # Start from the full original payload so output preserves all input fields.
        payload = source_payload

    prepare_results_rows(payload)

    payload["metadata"]["leetspeak_run"] = {
        "model": args.model,
        "input_path": str(args.input),
        "output_path": str(args.output),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system": args.system,
        "prompt_transform": 'remove trailing "." then append " but in leetspeak"',
        "total_prompts": len(payload["results"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_results(args.output, payload)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    to_process = [
        row
        for row in payload["results"]
        if isinstance(row, dict)
        and row.get("leetspeak_response") is None
        and row.get("leetspeak_error") is None
    ]
    already_processed = len(payload["results"]) - len(to_process)
    print(
        f"Loaded {len(payload['results'])} prompts, {already_processed} already processed, "
        f"{len(to_process)} remaining."
    )

    for idx, item in enumerate(to_process, start=1):
        subject = str(item.get("subject", ""))
        prompt = str(item.get("leetspeak_prompt", "")).strip()
        if not prompt:
            original_prompt = str(item.get("prompt", "")).strip()
            prompt = transform_prompt(original_prompt)
            item["leetspeak_prompt"] = prompt
        print(f"[{idx}/{len(to_process)}] {subject}")

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{args.system}\n"
                            'Begin your response with "<think>\\n".'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=False,
            )
            item["leetspeak_response"] = response.choices[0].message.content
            item["leetspeak_error"] = None
        except Exception as exc:
            item["leetspeak_error"] = str(exc)
            item["leetspeak_response"] = None
        item["leetspeak_model"] = args.model
        item["leetspeak_created_at"] = datetime.now(timezone.utc).isoformat()

        payload["metadata"]["leetspeak_run"]["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        payload["metadata"]["leetspeak_run"]["completed"] = sum(
            1
            for row in payload["results"]
            if isinstance(row, dict)
            and (row.get("leetspeak_response") is not None or row.get("leetspeak_error") is not None)
        )
        save_results(args.output, payload)

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"Done. Results saved incrementally to '{args.output}'.")


if __name__ == "__main__":
    main()
