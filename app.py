from flask import Flask, render_template, request, jsonify
import os
import re

try:
    import openai
except Exception:
    openai = None

app = Flask(__name__)


def preprocess_question(q: str):
    q_lower = q.lower()
    q_clean = re.sub(r"[^a-z0-9\s]", "", q_lower)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    tokens = q_clean.split(" ") if q_clean else []
    return q_clean, tokens


def build_prompt(processed_q: str) -> str:
    return (
        "You are a helpful assistant.\n"
        f"User question (preprocessed): {processed_q}\n"
        "Please answer concisely and clearly."
    )


def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set on server.")
    if openai is None:
        raise RuntimeError("openai package not installed on server.")

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"].strip()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.form or request.json
    question = data.get("question") if data else None
    if not question:
        return jsonify({"error": "No question provided"}), 400

    processed, tokens = preprocess_question(question)
    prompt = build_prompt(processed)

    try:
        answer = call_llm(prompt)
    except Exception as e:
        return jsonify({
            "question": question,
            "processed_question": processed,
            "prompt": prompt,
            "error": str(e),
        }), 500

    return jsonify({
        "question": question,
        "processed_question": processed,
        "prompt": prompt,
        "answer": answer,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
