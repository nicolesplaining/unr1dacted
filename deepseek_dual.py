#!/usr/bin/env python3
import argparse
import re
from typing import Any, Dict, List, Optional

from transformers import pipeline


MODEL_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


# def _strip_think(text: str) -> str:
#     # Remove any <think>...</think> blocks if the model emits them.
#     return re.sub(r"<think>.*?<\/think>", "", text, flags=re.DOTALL).strip()


def _extract_generated_text(result: List[Dict[str, Any]]) -> str:
    if not result:
        return ""
    generated = result[0].get("generated_text", "")
    if isinstance(generated, list):
        # Chat-style output: list of messages
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
        # Fallback: last item content if role is missing
        last = generated[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(generated)


def _build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # As per official recommendations for DeepSeek-R1 series to ensure thorough reasoning,
    # we start the assistant response with "<think>\n".
    messages.append({"role": "assistant", "content": "<think>\n"})
    return messages


def _run_model(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    trust_remote_code: bool,
    system_prompt: Optional[str],
) -> str:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    messages = _build_messages(prompt, system_prompt)
    result = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    text = _extract_generated_text(result)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompts against two DeepSeek R1 Distill Qwen models."
    )
    parser.add_argument("prompt", help="User prompt to send to both models.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for deterministic).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (otherwise greedy decoding).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the hub.",
    )
    parser.add_argument(
        "--rationale",
        action="store_true",
        help="Also ask for a brief high-level rationale (non-step-by-step).",
    )
    parser.add_argument(
        "--include-32b",
        action="store_true",
        help="Also run against the DeepSeek-R1-Distill-Qwen-32B model.",
    )
    args = parser.parse_args()

    system_prompt = (
        "You are a helpful assistant. Provide concise answers. "
        "Do not reveal chain-of-thought. If asked for reasoning, "
        "give a short, high-level rationale without step-by-step details."
    )

    print("== DeepSeek R1 Distill Qwen 7B ==")
    response_7b = _run_model(
        MODEL_7B,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.do_sample,
        args.trust_remote_code,
        system_prompt,
    )
    print(response_7b)

    if args.rationale:
        rationale_7b = _run_model(
            MODEL_7B,
            "Provide a brief rationale in 1-2 sentences (no step-by-step). "
            f"Question: {args.prompt}",
            512,
            args.temperature,
            args.do_sample,
            args.trust_remote_code,
            system_prompt,
        )
        print("\nRationale:", rationale_7b)

    print("\n== DeepSeek R1 Distill Qwen 1.5B ==")
    response_1_5b = _run_model(
        MODEL_1_5B,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.do_sample,
        args.trust_remote_code,
        system_prompt,
    )
    print(response_1_5b)

    if args.rationale:
        rationale_1_5b = _run_model(
            MODEL_1_5B,
            "Provide a brief rationale in 1-2 sentences (no step-by-step). "
            f"Question: {args.prompt}",
            512,
            args.temperature,
            args.do_sample,
            args.trust_remote_code,
            system_prompt,
        )
        print("\nRationale:", rationale_1_5b)

    if args.include_32b:
        print("\n== DeepSeek R1 Distill Qwen 32B ==")
        response_32b = _run_model(
            MODEL_32B,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.do_sample,
            args.trust_remote_code,
            system_prompt,
        )
        print(response_32b)

        if args.rationale:
            rationale_32b = _run_model(
                MODEL_32B,
                "Provide a brief rationale in 1-2 sentences (no step-by-step). "
                f"Question: {args.prompt}",
                512,
                args.temperature,
                args.do_sample,
                args.trust_remote_code,
                system_prompt,
            )
            print("\nRationale:", rationale_32b)


if __name__ == "__main__":
    main()
