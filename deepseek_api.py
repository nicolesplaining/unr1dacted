#!/usr/bin/env python3
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Call the DeepSeek API using the OpenAI-compatible client.")
    parser.add_argument("prompt", help="The prompt to send to the model.")
    parser.add_argument("--model", default="deepseek-chat", help="The DeepSeek model to use (e.g., deepseek-chat, deepseek-reasoner).")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt.")

    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it in your environment or in a .env file.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=False
        )

        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")

if __name__ == "__main__":
    main()
