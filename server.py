from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

# Load model
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("nllb_train_results")
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained(
    # "facebook/nllb-200-distilled-600M")
    # "facebook/nllb-200-3.3B")
    "nllb_train_results")
model.to("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=5555)
args = parser.parse_args()
app = Flask(__name__)


@app.route('/translate', methods=['POST'])
def api():
    sentence = request.form.get('message')
    u_id = request.form.get('id')

    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], max_length=300
    )
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    answer = translation
    return jsonify({'message': answer, 'id': u_id})

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(port=args.port, host=args.host)