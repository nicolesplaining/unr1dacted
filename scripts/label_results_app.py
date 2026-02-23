#!/usr/bin/env python3
"""Interactive result labeler for DeepSeek outputs."""

import json
from pathlib import Path
from typing import Any

import streamlit as st


INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results.json")
OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed.json")
LABEL_OPTIONS = ("type_1", "type_2", "type_3", "other")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _prepare_output_payload() -> dict[str, Any]:
    if OUTPUT_PATH.exists():
        payload = _read_json(OUTPUT_PATH)
    else:
        payload = _read_json(INPUT_PATH)

    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError("Expected a top-level 'results' list in the JSON payload.")

    # Keep original payload as-is and only add `type` key to each result item.
    for row in payload["results"]:
        if isinstance(row, dict) and "type" not in row:
            row["type"] = None

    _write_json(OUTPUT_PATH, payload)
    return payload


def _first_unlabeled_index(results: list[dict[str, Any]]) -> int:
    for i, row in enumerate(results):
        if row.get("type") is None:
            return i
    return max(0, len(results) - 1)


def _render_record(row: dict[str, Any], index: int, total: int) -> None:
    subject = row.get("subject", "")
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    error = row.get("error", None)
    current_type = row.get("type", None)

    st.markdown(f"### Record {index + 1} / {total}")
    st.markdown(f"**Current label:** `{current_type}`")
    st.markdown(f"**Subject**\n\n{subject}")
    st.markdown("**Prompt**")
    st.markdown(prompt)
    st.markdown("**Response**")
    st.markdown(response if response else "_(empty response)_")
    if error:
        st.markdown("**Error**")
        st.code(str(error))


def main() -> None:
    st.set_page_config(page_title="DeepSeek Result Labeler", layout="wide")
    st.title("DeepSeek Result Labeler")
    st.caption(
        "Shows one result at a time. Click a label to save immediately to "
        "`results/deepseek_r1_ccp_sensitive_results_typed.json`."
    )

    payload = _prepare_output_payload()
    results = payload["results"]
    if not results:
        st.warning("No results found.")
        return

    labeled = sum(1 for r in results if r.get("type") is not None)
    total = len(results)
    st.progress(labeled / total if total else 0.0)
    st.write(f"Labeled: **{labeled}/{total}**")

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = _first_unlabeled_index(results)

    # Keep index in range if payload changes.
    st.session_state.current_idx = max(0, min(st.session_state.current_idx, total - 1))

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("Previous", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.rerun()
    with col_next:
        if st.button("Next", use_container_width=True):
            st.session_state.current_idx = min(total - 1, st.session_state.current_idx + 1)
            st.rerun()

    current_idx = st.session_state.current_idx
    row = results[current_idx]
    _render_record(row, current_idx, total)

    st.markdown("### Choose label")
    btn_cols = st.columns(len(LABEL_OPTIONS))
    for col, label in zip(btn_cols, LABEL_OPTIONS):
        with col:
            if st.button(label, use_container_width=True):
                results[current_idx]["type"] = label
                _write_json(OUTPUT_PATH, payload)

                next_idx = current_idx + 1
                while next_idx < total and results[next_idx].get("type") is not None:
                    next_idx += 1
                st.session_state.current_idx = min(next_idx, total - 1)
                st.rerun()

    if labeled == total:
        st.success("All records are labeled.")


if __name__ == "__main__":
    main()
