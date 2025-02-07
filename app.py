from flask import Flask, request, jsonify
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return jsonify({"translated_text": translated_text[0]})

if __name__ == "__main__":
    app.run(debug=True)
