#!/usr/bin/env python3
"""
Run perplexity-ai/r1-1776 on prompts from an existing results JSON.

Input rows are preserved exactly and augmented with:
  - r1_1776_response

Writes output incrementally after each prompt.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import cache_utils
from transformers.utils import import_utils


MODEL_ID = "perplexity-ai/r1-1776"
DEFAULT_INPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak.json")
DEFAULT_OUTPUT_PATH = Path("results/deepseek_r1_ccp_sensitive_results_typed_cleaned_leetspeak_r1_1776.json")
DEFAULT_THINK_PREFILL = "<think>\n"


# Compatibility shim for model remote code that expects this helper in older transformers.
if not hasattr(import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available() -> bool:
        return False

    import_utils.is_torch_fx_available = _is_torch_fx_available


# Compatibility shim for remote code expecting DynamicCache.seen_tokens.
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "seen_tokens"):
    def _get_seen_tokens(self: Any) -> int:
        return int(getattr(self, "_seen_tokens", 0))

    def _set_seen_tokens(self: Any, value: int) -> None:
        self._seen_tokens = int(value)

    cache_utils.DynamicCache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)


# Compatibility shim for remote code expecting DynamicCache.get_max_length.
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_max_length"):
    def _get_max_length(self: Any) -> int | None:
        get_max_cache_shape = getattr(self, "get_max_cache_shape", None)
        if callable(get_max_cache_shape):
            max_shape = get_max_cache_shape()
            if isinstance(max_shape, int):
                return max_shape
            if isinstance(max_shape, (list, tuple)) and max_shape:
                last_dim = max_shape[-1]
                if isinstance(last_dim, int):
                    return last_dim
        for attr in ("max_length", "_max_length"):
            value = getattr(self, attr, None)
            if isinstance(value, int):
                return value
        return None

    cache_utils.DynamicCache.get_max_length = _get_max_length


# Compatibility shim for remote code expecting DynamicCache.get_seq_length.
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_seq_length"):
    def _get_seq_length(self: Any, layer_idx: int = 0) -> int:
        key_cache = getattr(self, "key_cache", None)
        if isinstance(key_cache, (list, tuple)) and key_cache:
            safe_idx = min(max(layer_idx, 0), len(key_cache) - 1)
            layer = key_cache[safe_idx]
            shape = getattr(layer, "shape", None)
            if shape is not None and len(shape) >= 2:
                return int(shape[-2])
        return int(getattr(self, "seen_tokens", 0))

    cache_utils.DynamicCache.get_seq_length = _get_seq_length


# Compatibility shim for remote code expecting DynamicCache.get_usable_length.
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
    def _get_usable_length(self: Any, new_seq_length: int, layer_idx: int = 0) -> int:
        previous_seq_length = int(self.get_seq_length(layer_idx))
        max_length = self.get_max_length()
        if max_length is None:
            return previous_seq_length
        if previous_seq_length + new_seq_length > max_length:
            return max(max_length - new_seq_length, 0)
        return previous_seq_length

    cache_utils.DynamicCache.get_usable_length = _get_usable_length


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


def should_process(response: Any, retry_errors: bool) -> bool:
    if response is None:
        return True
    if isinstance(response, str) and response.strip() == "":
        return True
    if retry_errors and isinstance(response, str) and response.startswith("ERROR: "):
        return True
    return False


def pick_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_with_fallbacks(model_id: str, torch_dtype: torch.dtype) -> Any:
    # Attempt 1: auto-shard with accelerate hooks.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=False,
            attn_implementation="eager",
            offload_buffers=True,
        )
        model.eval()
        print("Loaded model with device_map=auto.")
        return model
    except Exception as exc:
        print(f"Auto device_map load failed: {type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Attempt 2: plain load without accelerate dispatch, then move to single device.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    )
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()
    print(f"Loaded model on single device: {target_device}.")
    return model


def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    use_cache: bool,
    add_think_prefill: bool,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if add_think_prefill:
        messages.append({"role": "assistant", "content": DEFAULT_THINK_PREFILL})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_device = get_model_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": use_cache,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    if add_think_prefill and text and not text.startswith(DEFAULT_THINK_PREFILL.strip()):
        text = DEFAULT_THINK_PREFILL + text
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run r1-1776 on original prompts in a results JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Input JSON (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help=f"Output JSON (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature when --do-sample is enabled.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument(
        "--with-think-prefill",
        action="store_true",
        help="Opt in to assistant '<think>\\n' prefill (disabled by default).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable KV cache during generation (disabled by default for stability).",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Also rerun rows where r1_1776_response starts with 'ERROR: '.",
    )
    args = parser.parse_args()

    payload = load_payload(args.input)
    if args.output.exists():
        output_payload = load_payload(args.output)
        input_results = payload.get("results", [])
        output_results = output_payload.get("results", [])
        if isinstance(input_results, list) and isinstance(output_results, list) and len(input_results) == len(output_results):
            payload = output_payload
            for idx, src_row in enumerate(input_results):
                dst_row = output_results[idx]
                if isinstance(src_row, dict) and isinstance(dst_row, dict):
                    for key, value in src_row.items():
                        dst_row.setdefault(key, value)

    results = payload["results"]
    for row in results:
        if isinstance(row, dict) and "r1_1776_response" not in row:
            row["r1_1776_response"] = None
    save_payload(args.output, payload)

    print(f"Loading model: {MODEL_ID}")
    torch_dtype = pick_torch_dtype()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_model_with_fallbacks(MODEL_ID, torch_dtype)

    pending: list[int] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        if should_process(row.get("r1_1776_response"), args.retry_errors):
            pending.append(idx)
    print(f"Total rows: {len(results)} | Pending: {len(pending)}")

    for n, idx in enumerate(pending, start=1):
        row = results[idx]
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            row["r1_1776_response"] = ""
            save_payload(args.output, payload)
            continue

        try:
            row["r1_1776_response"] = generate_one(
                tokenizer,
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                use_cache=args.use_cache,
                add_think_prefill=args.with_think_prefill,
            )
        except Exception as exc:
            row["r1_1776_response"] = f"ERROR: {type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_payload(args.output, payload)
        print(f"[{n}/{len(pending)}] saved row {idx + 1}")

    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
