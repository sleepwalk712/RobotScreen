import json
import os
# seems wrong, but for some reason manually invoking garbage collection
# is necessary to release memory after predictions (?)
import gc

import numpy as np
from flask import Flask, request, jsonify
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader

import screening_model
from screening_model import SRDataset
import config

WEIGHTS_PATH = config.WEIGHTS_PATH


app = Flask(__name__)

# for now, assuming predictions on cpu.
device = "cpu"


@app.route('/')
def hello():
    return 'Welcome to RobotScreener ;)'


@app.route('/train/abstract_screenings/<uuid>', methods=['POST'])
def train(uuid: str):
    # studies = json.loads(request.json)['articles']
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
    # studies = json.loads(request.json)['input_citations']
    unlabel_data = request.json['input_citations']
    timestamp = request.json['timestamp']

    titles, abstracts = [], []

    for citation in unlabel_data:
        titles.append(citation['ti'])
        abstracts.append(citation['abs'])

    dataset = SRDataset(titles, abstracts)

    # we just outright assume that we are using Roberta; this will break
    # if untrue. TODO probably want to add flexibility here.
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")
    model = RobertaForSequenceClassification.from_pretrained(
        "allenai/biomed_roberta_base",
        num_labels=2,
    ).to(device=device)

    # note that we assume a *.pt extension for the pytorch stuff.
    file_name = f"abstract_screening_{uuid}_{timestamp}.pt"
    weights_path = os.path.join(WEIGHTS_PATH, file_name)
    print(f"loading model weights from {weights_path}...")
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device(device)))

    dl = DataLoader(dataset, batch_size=8)
    preds, _ = screening_model.make_preds(dl, model, tokenizer, device=device)

    # oddly without this memory will not be released following the predictions
    gc.collect()
    return jsonify({"predictions": preds})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
