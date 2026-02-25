#!/usr/bin/env python3
"""
Run Perplexity Sonar API on prompts from a results JSON and save incrementally.

Input payload is expected to be:
{
  "results": [
    {"subject": "...", "prompt": "...", ...},
    ...
  ]
}

Each row is preserved and augmented with:
  - sonar_response
  - sonar_error
  - sonar_model
  - sonar_created_at
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
    def load_dotenv() -> bool:
        return False


DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak_sonar.json")
DEFAULT_MODEL = "sonar-reasoning-pro"
BASE_URL = "https://api.perplexity.ai"


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        payload["metadata"] = {}
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def should_process(row: dict[str, Any], retry_errors: bool) -> bool:
    response = row.get("sonar_response")
    error = row.get("sonar_error")
    if response is None:
        return True
    if isinstance(response, str) and response.strip() == "":
        return True
    if retry_errors and isinstance(error, str) and error.strip():
        return True
    return False


def get_api_key() -> str:
    load_dotenv()
    key = (
        os.getenv("PPLX_API_KEY")
        or os.getenv("PERPLEXITY_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing API key. Set PPLX_API_KEY (preferred) or PERPLEXITY_API_KEY."
        )
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Perplexity Sonar API runner with incremental JSON writes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Perplexity Sonar model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt for all calls.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry rows that currently have sonar_error set.",
    )
    args = parser.parse_args()

    payload = load_payload(args.input)
    if args.output.exists():
        existing = load_payload(args.output)
        in_results = payload.get("results", [])
        out_results = existing.get("results", [])
        if isinstance(in_results, list) and isinstance(out_results, list) and len(in_results) == len(out_results):
            payload = existing
            for idx, src_row in enumerate(in_results):
                dst_row = out_results[idx]
                if isinstance(src_row, dict) and isinstance(dst_row, dict):
                    for key, value in src_row.items():
                        dst_row.setdefault(key, value)

    results = payload["results"]
    for row in results:
        if isinstance(row, dict):
            row.setdefault("sonar_response", None)
            row.setdefault("sonar_error", None)
            row.setdefault("sonar_model", None)
            row.setdefault("sonar_created_at", None)

    payload["metadata"]["sonar_run"] = {
        "provider": "perplexity",
        "base_url": BASE_URL,
        "model": args.model,
        "input_path": str(args.input),
        "output_path": str(args.output),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system": args.system,
        "total_prompts": len(results),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_payload(args.output, payload)

    api_key = get_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    pending: list[int] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        if should_process(row, args.retry_errors):
            pending.append(idx)
    print(f"Total rows: {len(results)} | Pending: {len(pending)}")

    for n, idx in enumerate(pending, start=1):
        row = results[idx]
        prompt = str(row.get("prompt", "")).strip()
        subject = str(row.get("subject", ""))
        print(f"[{n}/{len(pending)}] {subject}")

        if not prompt:
            row["sonar_response"] = ""
            row["sonar_error"] = "Empty prompt."
            row["sonar_model"] = args.model
            row["sonar_created_at"] = datetime.now(timezone.utc).isoformat()
            save_payload(args.output, payload)
            continue

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=False,
            )
            row["sonar_response"] = response.choices[0].message.content
            row["sonar_error"] = None
        except Exception as exc:
            row["sonar_response"] = None
            row["sonar_error"] = f"{type(exc).__name__}: {exc}"

        row["sonar_model"] = args.model
        row["sonar_created_at"] = datetime.now(timezone.utc).isoformat()
        payload["metadata"]["sonar_run"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["metadata"]["sonar_run"]["completed"] = sum(
            1
            for item in results
            if isinstance(item, dict)
            and (item.get("sonar_response") is not None or item.get("sonar_error") is not None)
        )
        save_payload(args.output, payload)

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"Done. Results saved incrementally to '{args.output}'.")


if __name__ == "__main__":
    main()
