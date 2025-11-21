"""
LLM_QA_CLI.py
Simple CLI that accepts a natural-language question, preprocesses it,
constructs a prompt and sends it to the OpenAI API (ChatCompletion).

Usage:
  python LLM_QA_CLI.py            # interactive: prompts for a question
  python LLM_QA_CLI.py "What is GPT?"  # question as argument

Environment:
  export OPENAI_API_KEY=your_api_key_here
  (or create a .env file with OPENAI_API_KEY)

Notes: If you prefer another provider, replace the `call_llm` implementation.
"""

import os
import sys
import re
import json
from typing import Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import openai
except Exception:
    openai = None


def preprocess_question(q: str) -> Tuple[str, list]:
    q_lower = q.lower()
    q_clean = re.sub(r"[^a-z0-9\s]", "", q_lower)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    tokens = q_clean.split(" ") if q_clean else []
    return q_clean, tokens


def build_prompt(processed_q: str) -> str:
    return (
        "You are a helpful, concise assistant. \n"
        f"User question (preprocessed): {processed_q}\n"
        "Please provide a clear, short answer and include a brief explanation if relevant."
    )


def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    if openai is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")

    openai.api_key = api_key

    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    return resp["choices"][0]["message"]["content"].strip()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    print("\n[1] Original Question:\n", question)
    processed, tokens = preprocess_question(question)
    print("\n[2] Processed Question:\n", processed)
    print("\n[3] Tokens (first 50):\n", tokens[:50])

    prompt = build_prompt(processed)
    print("\n[4] Prompt sent to LLM:\n", prompt)

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("\nError when calling LLM:\n", e)
        return

    print("\n[5] LLM Answer:\n", answer)


if __name__ == "__main__":
    main()
