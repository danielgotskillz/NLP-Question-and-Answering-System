from flask import Flask, render_template, request
import google.generativeai as genai
import re

app = Flask(__name__)
genai.configure(api_key="AIzaSyCg4AmnKeXAUHFGzoYh6WQ3Q-KnnrJbh84")

def preprocess(question):
    question = question.lower()
    question = re.sub(r'[^a-z0-9\s]', '', question)
    tokens = question.split()
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def home():
    processed = ""
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        processed = preprocess(question)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(question)
        answer = response.text
    return render_template("index.html", processed=processed, answer=answer)

if __name__ == "__main__":
    app.run(port=5000)
