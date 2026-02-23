#!/usr/bin/env python3
"""
Run DeepSeek-R1 (API) on a prompt JSON file and save results incrementally.

Input JSON format:
[
  {"subject": "...", "prompt": "..."},
  ...
]
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


DEFAULT_INPUT_PATH = Path("data/ccp_sensitive_prompts.json")
DEFAULT_OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results.json")


def load_prompts(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in '{path}', got: {type(data).__name__}")

    prompts: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not an object.")
        subject = str(item.get("subject", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not subject or not prompt:
            raise ValueError(
                f"Item at index {i} must contain non-empty 'subject' and 'prompt'."
            )
        prompts.append({"subject": subject, "prompt": prompt})
    return prompts


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


def build_processed_keys(results: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for item in results:
        subject = str(item.get("subject", ""))
        prompt = str(item.get("prompt", ""))
        if subject and prompt:
            keys.add((subject, prompt))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch DeepSeek-R1 API runner with incremental JSON writes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input prompts JSON path (default: {DEFAULT_INPUT_PATH})",
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

    prompts = load_prompts(args.input)
    payload = load_existing_results(args.output)
    processed = build_processed_keys(payload["results"])

    payload["metadata"] = {
        "model": args.model,
        "input_path": str(args.input),
        "output_path": str(args.output),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system": args.system,
        "total_prompts": len(prompts),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_results(args.output, payload)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    to_process = [p for p in prompts if (p["subject"], p["prompt"]) not in processed]
    print(
        f"Loaded {len(prompts)} prompts, {len(processed)} already processed, "
        f"{len(to_process)} remaining."
    )

    for idx, item in enumerate(to_process, start=1):
        subject = item["subject"]
        prompt = item["prompt"]
        print(f"[{idx}/{len(to_process)}] {subject}")

        record: dict[str, Any] = {
            "subject": subject,
            "prompt": prompt,
            "response": None,
            "error": None,
            "model": args.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

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
            record["response"] = response.choices[0].message.content
        except Exception as exc:
            record["error"] = str(exc)

        payload["results"].append(record)
        payload["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["metadata"]["completed"] = len(payload["results"])
        save_results(args.output, payload)

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"Done. Results saved incrementally to '{args.output}'.")


if __name__ == "__main__":
    main()
