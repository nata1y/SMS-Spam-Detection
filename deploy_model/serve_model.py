"""
Flask API of the SMS Spam detection model model.
"""
import monitoring.serve_metrics

import pandas as pd

from flask import Flask, jsonify, request
from flasgger import Swagger

from deploy_model.drift_manager import DriftManager
from deploy_model.util import load_best_clf
from deploy_model.util import ensure_path_exists
from train_model.text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)
classifier_name = None
ensure_path_exists('output/stats')
stats = None
manager = DriftManager()


try:
    stats = pd.read_csv('output/stats/stats_from_wild.csv')
except Exception as e:
    print(e)
    stats = pd.DataFrame([], columns=["result", "prob_spam", "classifier", "sms"])
    stats.to_csv('output/stats/stats_from_wild.csv', index=False)


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
    global stats
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    model, _ = load_best_clf()
    prediction = model.predict(processed_sms)[0]

    stats = stats.append({
                    "result": prediction,
                    "prob_spam": model.predict_proba(processed_sms)[0],
                    "classifier": classifier_name,
                    "sms": sms
                }, ignore_index=True)
    stats.to_csv('output/stats/stats_from_wild.csv', index=False)

    manager.add_call([prediction, sms])

    return jsonify({
        "result": prediction,
        "classifier": classifier_name,
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
        "classifier": classifier_name,
        "sms": sms
    })


if __name__ == '__main__':
    clf, classifier_name = load_best_clf()
    app.run(host="0.0.0.0", port=8080, debug=True)
