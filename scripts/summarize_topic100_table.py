#!/usr/bin/env python3
"""
Summarize filtered results using a per-topic 100-total normalization.

For each subject:
1) Scale baseline counts so type_1 + type_2 = 100 (integer feasible split).
2) Compute matched counts in that same scaled space (integer feasible counts).
3) Report percentages from the scaled numerators/denominators.
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_BASELINE_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned.json")
DEFAULT_FIELD = "response"
CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SUBJECT_ORDER = [
    "Communist Party of China",
    "Hong Kong Autonomy",
    "Taiwan Independence",
    "Tiananmen Square incident",
    "Xi Jinping",
    "Xinjiang Uyghurs",
]


def integer_split_100(part_a: float, part_b: float) -> tuple[int, int]:
    """Return integer counts that sum to 100 and best match two proportions."""
    total = part_a + part_b
    if total <= 0:
        return 0, 0
    ideal_a = 100.0 * part_a / total
    a = int(round(ideal_a))
    a = max(0, min(100, a))
    b = 100 - a
    return a, b


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    return payload


def compute_type_counts(results: list[Any]) -> "OrderedDict[str, dict[str, int]]":
    counts: "OrderedDict[str, dict[str, int]]" = OrderedDict()
    for row in results:
        if not isinstance(row, dict):
            continue
        subject = str(row.get("subject", "")).strip()
        row_type = str(row.get("type", "")).strip()
        if not subject or row_type not in {"type_1", "type_2"}:
            continue
        if subject not in counts:
            counts[subject] = {"type_1": 0, "type_2": 0}
        counts[subject][row_type] += 1
    return counts


def response_matches(response: str, mode: str, needle: str, ignore_case: bool) -> bool:
    if mode == "all":
        return True
    if mode == "chinese":
        return bool(CHINESE_CHAR_RE.search(response))
    if mode == "substring":
        if ignore_case:
            return needle.lower() in response.lower()
        return needle in response
    raise ValueError(f"Unknown mode: {mode}")


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("Subject", "t1_100", "t2_100", "match_t1_100", "match_t2_100", "pct_t1", "pct_t2", "pct_all")
    ws = max(len(headers[0]), *(len(str(r["subject"])) for r in rows))
    w = [max(len(h), 10) for h in headers[1:]]

    sep = (
        f"+-{'-' * ws}-+-{'-' * w[0]}-+-{'-' * w[1]}-+-{'-' * w[2]}-+-{'-' * w[3]}-+-{'-' * w[4]}-+-{'-' * w[5]}-+-{'-' * w[6]}-+"
    )
    print(sep)
    print(
        f"| {headers[0].ljust(ws)} | {headers[1].rjust(w[0])} | {headers[2].rjust(w[1])} | {headers[3].rjust(w[2])} | {headers[4].rjust(w[3])} | {headers[5].rjust(w[4])} | {headers[6].rjust(w[5])} | {headers[7].rjust(w[6])} |"
    )
    print(sep)
    for r in rows:
        print(
            f"| {str(r['subject']).ljust(ws)} | "
            f"{r['t1_100']:>{w[0]}.1f} | "
            f"{r['t2_100']:>{w[1]}.1f} | "
            f"{r['match_t1_100']:>{w[2]}.1f} | "
            f"{r['match_t2_100']:>{w[3]}.1f} | "
            f"{r['pct_t1']:>{w[4]}.1f} | "
            f"{r['pct_t2']:>{w[5]}.1f} | "
            f"{r['pct_all']:>{w[6]}.1f} |"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-topic 100-normalized summary table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--baseline-input", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--field", type=str, default=DEFAULT_FIELD)
    parser.add_argument("--mode", choices=["all", "chinese", "substring"], default="all")
    parser.add_argument("--needle", type=str, default="")
    parser.add_argument("--ignore-case", action="store_true")
    args = parser.parse_args()

    if args.mode == "substring" and not args.needle:
        raise ValueError("--needle is required when --mode substring is used.")

    input_payload = load_payload(args.input)
    baseline_payload = load_payload(args.baseline_input)

    baseline_counts = compute_type_counts(baseline_payload["results"])
    matched_counts: "OrderedDict[str, dict[str, int]]" = OrderedDict(
        (s, {"type_1": 0, "type_2": 0}) for s in baseline_counts.keys()
    )

    for row in input_payload["results"]:
        if not isinstance(row, dict):
            continue
        subject = str(row.get("subject", "")).strip()
        row_type = str(row.get("type", "")).strip()
        response = row.get(args.field)
        if subject not in matched_counts:
            continue
        if row_type not in {"type_1", "type_2"}:
            continue
        if not isinstance(response, str) or not response.strip():
            continue
        if not response_matches(response, args.mode, args.needle, args.ignore_case):
            continue
        matched_counts[subject][row_type] += 1

    rows: list[dict[str, Any]] = []
    for subject, base in baseline_counts.items():
        b1 = float(base["type_1"])
        b2 = float(base["type_2"])
        bt = b1 + b2
        m1 = float(matched_counts[subject]["type_1"])
        m2 = float(matched_counts[subject]["type_2"])
        if bt <= 0:
            rows.append(
                {
                    "subject": subject,
                    "t1_100": 0.0,
                    "t2_100": 0.0,
                    "match_t1_100": 0.0,
                    "match_t2_100": 0.0,
                    "pct_t1": 0.0,
                    "pct_t2": 0.0,
                    "pct_all": 0.0,
                }
            )
            continue

        t1_100_i, t2_100_i = integer_split_100(b1, b2)
        t1_100 = float(t1_100_i)
        t2_100 = float(t2_100_i)

        # Compute matched counts from scaled denominators so every value is
        # feasible in the 100-sample-per-topic interpretation.
        p1 = (m1 / b1) if b1 > 0 else 0.0
        p2 = (m2 / b2) if b2 > 0 else 0.0
        match_t1_100_i = int(round(p1 * t1_100_i))
        match_t2_100_i = int(round(p2 * t2_100_i))
        match_t1_100_i = max(0, min(t1_100_i, match_t1_100_i))
        match_t2_100_i = max(0, min(t2_100_i, match_t2_100_i))

        match_t1_100 = float(match_t1_100_i)
        match_t2_100 = float(match_t2_100_i)
        pct_t1 = (match_t1_100_i / t1_100_i * 100.0) if t1_100_i > 0 else 0.0
        pct_t2 = (match_t2_100_i / t2_100_i * 100.0) if t2_100_i > 0 else 0.0
        pct_all = (match_t1_100_i + match_t2_100_i)

        rows.append(
            {
                "subject": subject,
                "t1_100": t1_100,
                "t2_100": t2_100,
                "match_t1_100": match_t1_100,
                "match_t2_100": match_t2_100,
                "pct_t1": pct_t1,
                "pct_t2": pct_t2,
                "pct_all": pct_all,
            }
        )

    order_index = {subject: idx for idx, subject in enumerate(SUBJECT_ORDER)}
    rows.sort(key=lambda r: (order_index.get(str(r["subject"]), len(order_index)), str(r["subject"])))

    print_table(rows)


if __name__ == "__main__":
    main()
