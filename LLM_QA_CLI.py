import google.generativeai as genai
import re

# Configure Gemini API Key
genai.configure(api_key="AIzaSyCg4AmnKeXAUHFGzoYh6WQ3Q-KnnrJbh84")

def preprocess(question):
    question = question.lower()
    question = re.sub(r'[^a-z0-9\s]', '', question)
    tokens = question.split()
    return " ".join(tokens)

def ask_llm(question):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    return response.text

if __name__ == "__main__":
    while True:
        user_q = input("Enter your question (or 'quit'): ")
        if user_q.lower() == "quit":
            break
        processed = preprocess(user_q)
        print("Processed Question:", processed)
        answer = ask_llm(user_q)
        print("Answer:", answer)
