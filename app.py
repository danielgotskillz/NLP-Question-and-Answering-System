import os
import re
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)

client = OpenAI()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    processed = ""
    answer = ""

    if request.method == "POST":
        question = request.form.get("question")

        processed = preprocess_text(question)

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a helpful question-answering assistant."},
                {"role": "user", "content": processed}
            ]
        )

        answer = response.choices[0].message["content"]

    return render_template("index.html", processed=processed, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
