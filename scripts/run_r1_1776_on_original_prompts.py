#!/usr/bin/env python3
"""
Run perplexity-ai/r1-1776 on original prompts from an existing results JSON.

Input rows are preserved exactly and augmented with:
  - r1_1776_response

Writes output incrementally after each prompt.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import pipeline


MODEL_ID = "perplexity-ai/r1-1776"
DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak_r1_1776.json")


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def extract_assistant_text(result: list[dict[str, Any]]) -> str:
    if not result:
        return ""
    generated = result[0].get("generated_text", "")
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
        last = generated[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run r1-1776 on original prompts in a results JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Input JSON (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help=f"Output JSON (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    args = parser.parse_args()

    payload = load_payload(args.input)
    results = payload["results"]

    # Preserve all original fields and add one new key on each row.
    for row in results:
        if isinstance(row, dict) and "r1_1776_response" not in row:
            row["r1_1776_response"] = None

    save_payload(args.output, payload)

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )

    pending = [idx for idx, row in enumerate(results) if isinstance(row, dict) and row.get("r1_1776_response") is None]
    print(f"Total rows: {len(results)} | Pending: {len(pending)}")

    for n, idx in enumerate(pending, start=1):
        row = results[idx]
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            row["r1_1776_response"] = ""
            save_payload(args.output, payload)
            continue

        messages = [{"role": "user", "content": prompt}]
        try:
            out = pipe(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
            row["r1_1776_response"] = extract_assistant_text(out)
        except Exception as exc:
            row["r1_1776_response"] = f"ERROR: {exc}"

        save_payload(args.output, payload)
        print(f"[{n}/{len(pending)}] saved row {idx + 1}")

    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
