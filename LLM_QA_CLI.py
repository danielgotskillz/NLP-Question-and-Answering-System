import os
import re
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(tokens)

def ask_llm(question):
    processed = preprocess_text(question)

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "You are a helpful question-answering assistant."},
            {"role": "user", "content": processed}
        ]
    )
    
    return response.choices[0].message["content"]

if __name__ == "__main__":
    print("=== LLM Q&A CLI ===")
    while True:
        user_input = input("Enter your question (or 'quit'): ")
        if user_input.lower() == "quit":
            break

        answer = ask_llm(user_input)
        print("\nAnswer:", answer, "\n")
