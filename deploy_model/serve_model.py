"""
Flask API of the SMS Spam detection model model.
"""
import traceback
from datetime import datetime

import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import os

from train_model.text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)


def load_best_clf():
    file_to_load, latest_date = 'model.joblib', datetime.strptime('01-01-1970', "%m-%d-%Y")
    for filename in os.listdir('output'):
        if filename.endswith(".joblib"):
            try:
                if latest_date < datetime.strptime(filename.split('_')[1].split('.')[0], "%m-%d-%Y"):
                    file_to_load = filename
                    latest_date = datetime.strptime(filename.split('_')[1].split('.')[0], "%m-%d-%Y")
                    print(latest_date)
            except Exception as e:
                print(e)

    return joblib.load('output/' + file_to_load)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    model = load_best_clf()
    prediction = model.predict(processed_sms)[0]

    return jsonify({
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    })


@app.route('/dumbpredict', methods=['POST'])
def dumb_predict():
    """
    Predict whether a given SMS is Spam or Ham (dumb model: always predicts 'ham').
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')

    return jsonify({
        "result": "Spam",
        "classifier": "decision tree",
        "sms": sms
    })


if __name__ == '__main__':
    clf = load_best_clf()
    app.run(port=8080, debug=True)
