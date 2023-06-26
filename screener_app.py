import os
import gc

import numpy as np
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

import screening_model
from screening_model import SRDataset
import config


app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to RobotScreener ;)'


@app.route('/train/abstract_screenings/<uuid>', methods=['POST'])
def train(uuid: str):
    labeled_data = request.json['labeled_data']

    titles, abstracts, labels = [], [], []

    for citation in labeled_data:
        titles.append(citation['ti'])
        abstracts.append(citation['abs'])
        labels.append(int(citation['label']))

    dataset = SRDataset(titles, abstracts, np.array(labels))

    response = screening_model.train_and_save(
        dataset,
        uuid,
        batch_size=8,
        epochs=1,
    )
    return jsonify(response)


@app.route('/predict/abstract_screenings/<uuid>', methods=['POST'])
def predict(uuid: str):
    unlabel_data = request.json['input_citations']
    timestamp = request.json['timestamp']

    titles, abstracts = [], []

    for citation in unlabel_data:
        titles.append(citation['ti'])
        abstracts.append(citation['abs'])

    dataset = SRDataset(titles, abstracts)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2).to(device=config.DEVICE)

    file_name = f"abstract_screening_{uuid}_{timestamp}.pt"
    weights_path = os.path.join(config.WEIGHTS_PATH, file_name)
    print(f"loading model weights from {weights_path}...")
    try:
        model.load_state_dict(torch.load(
            weights_path, map_location=torch.device(config.DEVICE)))
    except FileNotFoundError:
        return jsonify({"error": "model weights not found"}), 404

    dl = DataLoader(dataset, batch_size=8)
    try:
        preds, _ = screening_model.make_preds(
            dl, model, tokenizer, device=config.DEVICE)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    gc.collect()
    return jsonify({"predictions": preds})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
