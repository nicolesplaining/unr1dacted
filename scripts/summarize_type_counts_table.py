#!/usr/bin/env python3
"""
Summarize type_1/type_2 counts by subject in a simple table.

Supports optional response filtering:
- all rows
- rows whose response contains Chinese characters
- rows whose response contains a substring
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


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level 'results' list in '{path}'.")
    return payload


def response_matches(
    response: str,
    mode: str,
    needle: str,
    ignore_case: bool,
) -> bool:
    if mode == "all":
        return True
    if mode == "chinese":
        return bool(CHINESE_CHAR_RE.search(response))
    if mode == "substring":
        if ignore_case:
            return needle.lower() in response.lower()
        return needle in response
    raise ValueError(f"Unknown mode: {mode}")


def print_table(
    rows: list[tuple[str, float, float, float]],
    value_label: str,
    decimals: int,
) -> None:
    headers = ("Subject", "t1", "t2", "tot")
    width_subject = max([len(headers[0])] + [len(subject) for subject, _, _, _ in rows])
    width_t1 = max(len(headers[1]), 6)
    width_t2 = max(len(headers[2]), 6)
    width_total = max(len(headers[3]), 6)

    sep = (
        f"+-{'-' * width_subject}-+-{'-' * width_t1}-+-{'-' * width_t2}-+-{'-' * width_total}-+"
    )
    print(f"value_mode: {value_label}")
    print(sep)
    print(
        f"| {headers[0].ljust(width_subject)} | {headers[1].rjust(width_t1)} | {headers[2].rjust(width_t2)} | {headers[3].rjust(width_total)} |"
    )
    print(sep)
    for subject, t1, t2, total in rows:
        t1_str = f"{t1:.{decimals}f}" if isinstance(t1, float) else str(t1)
        t2_str = f"{t2:.{decimals}f}" if isinstance(t2, float) else str(t2)
        total_str = f"{total:.{decimals}f}" if isinstance(total, float) else str(total)
        print(
            f"| {subject.ljust(width_subject)} | {t1_str.rjust(width_t1)} | {t2_str.rjust(width_t2)} | {total_str.rjust(width_total)} |"
        )
    print(sep)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print type_1/type_2 count table by subject."
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
        "--mode",
        choices=["all", "chinese", "substring"],
        default="all",
        help="How to filter rows before counting.",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default="",
        help='Substring for --mode substring (example: "4").',
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive substring match.",
    )
    parser.add_argument(
        "--scale-to",
        type=float,
        default=0.0,
        help=(
            "If > 0, output scaled values (example: 100). If 0, show raw counts."
        ),
    )
    parser.add_argument(
        "--use-baseline-total",
        action="store_true",
        help=(
            "When scaling, use per-subject totals from --baseline-input as the denominator "
            "(matches the graph totals), instead of this filtered subset total."
        ),
    )
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=(
            "Baseline JSON used for per-subject type_1+type_2 totals when "
            "--use-baseline-total is set."
        ),
    )
    parser.add_argument(
        "--separate-type-baselines",
        action="store_true",
        help=(
            "When used with --use-baseline-total and scaling, divide type_1 by baseline "
            "type_1 and type_2 by baseline type_2 (per subject), instead of using a single "
            "combined baseline total."
        ),
    )
    parser.add_argument(
        "--compute-from-scaled",
        action="store_true",
        help=(
            "When scaling, first round scaled counts, then report percentages from those "
            "scaled counts (useful for 100-based interpretation)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "substring" and not args.needle:
        raise ValueError("--needle is required when --mode substring is used.")

    payload = load_payload(args.input)
    results = payload["results"]

    counts: "OrderedDict[str, dict[str, int]]" = OrderedDict()
    included_rows = 0

    for row in results:
        if not isinstance(row, dict):
            continue
        subject = str(row.get("subject", "")).strip()
        row_type = str(row.get("type", "")).strip()
        response = row.get(args.field)

        if not subject or row_type not in {"type_1", "type_2"}:
            continue
        if not isinstance(response, str) or not response.strip():
            continue
        if not response_matches(response, args.mode, args.needle, args.ignore_case):
            continue

        if subject not in counts:
            counts[subject] = {"type_1": 0, "type_2": 0}
        counts[subject][row_type] += 1
        included_rows += 1

    table_rows: list[tuple[str, float, float, float]] = []
    baseline_counts: "OrderedDict[str, dict[str, int]]" = OrderedDict()
    subject_order: list[str]
    if args.use_baseline_total:
        baseline_payload = load_payload(args.baseline_input)
        baseline_counts = compute_type_counts(baseline_payload["results"])
        subject_order = list(baseline_counts.keys())
    else:
        subject_order = list(counts.keys())

    for subject in subject_order:
        c = counts.get(subject, {"type_1": 0, "type_2": 0})
        t1 = float(c["type_1"])
        t2 = float(c["type_2"])
        total = t1 + t2
        if args.scale_to > 0:
            if args.use_baseline_total and args.separate_type_baselines:
                baseline = baseline_counts.get(subject, {"type_1": 0, "type_2": 0})
                b1 = float(baseline["type_1"])
                b2 = float(baseline["type_2"])
                t1 = (t1 / b1 * args.scale_to) if b1 > 0 else 0.0
                t2 = (t2 / b2 * args.scale_to) if b2 > 0 else 0.0
                if args.compute_from_scaled:
                    t1 = round(t1)
                    t2 = round(t2)
                total = t1 + t2
            else:
                if args.use_baseline_total:
                    baseline_total = float(
                        baseline_counts.get(subject, {"type_1": 0, "type_2": 0})["type_1"]
                        + baseline_counts.get(subject, {"type_1": 0, "type_2": 0})["type_2"]
                    )
                    denom = baseline_total if baseline_total > 0 else 0.0
                else:
                    denom = total

                if denom > 0:
                    factor = args.scale_to / denom
                else:
                    factor = 0.0
                t1 *= factor
                t2 *= factor
                if args.compute_from_scaled:
                    t1 = round(t1)
                    t2 = round(t2)
                total = t1 + t2
        table_rows.append((subject, t1, t2, total))

    if not table_rows:
        print("No rows matched the selected filters.")
        return

    label = "count"
    if args.scale_to > 0:
        scaled_to = int(args.scale_to) if args.scale_to.is_integer() else args.scale_to
        if args.use_baseline_total and args.separate_type_baselines:
            label = f"scaled_to_{scaled_to} (separate type baselines)"
        elif args.use_baseline_total:
            label = f"scaled_to_{scaled_to} (baseline total)"
        else:
            label = f"scaled_to_{scaled_to}"
    decimals = 1 if args.scale_to > 0 else 0
    if args.scale_to > 0 and args.compute_from_scaled:
        label += ", computed_from_rounded_scaled"
    print_table(table_rows, label, decimals)
    print(f"Included rows: {included_rows}")


if __name__ == "__main__":
    main()
