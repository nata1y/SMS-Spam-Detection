# pylint: disable=W0603, W0611
"""
Flask API of the SMS Spam detection model model.
"""
import pandas as pd

from flask import Flask, jsonify, request
from flasgger import Swagger

from monitoring.MetricsManager import MetricsManager
from deploy_model.drift_manager import DriftManager
from deploy_model.util import load_best_clf
from deploy_model.util import ensure_path_exists
from train_model.text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)
ensure_path_exists('output/stats')

metricsManager: MetricsManager = MetricsManager()
manager = DriftManager(metricsManager)
STATS = None

try:
    STATS = pd.read_csv('output/stats/stats_from_wild.csv')
except FileNotFoundError:
    STATS = pd.DataFrame([], columns=["result", "prob_spam", "classifier", "sms"])
    STATS.to_csv('output/stats/stats_from_wild.csv', index=False)


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
    global STATS
    input_data = request.get_json()
    sms = input_data.get('sms')
    drift_type = input_data.get('drift_type')
    processed_sms = prepare(sms)
    model, model_name = load_best_clf()
    if drift_type in ['VANILLA_TRAINING', 'VANILLA_INCOMING']:
        prediction = model.predict(processed_sms)[0]
    else:
        prediction = input_data.get('label')
        window_size = input_data.get('window_size')
        manager.set_window_size(window_size)

    real_label = input_data.get('real_label')
    manager.add_real_label(real_label)

    STATS = STATS.append({
                    "result": prediction,
                    "prob_spam": model.predict_proba(processed_sms)[0],
                    "classifier": model_name,
                    "sms": sms
                }, ignore_index=True)
    STATS.to_csv('output/stats/stats_from_wild.csv', index=False)

    manager.add_call([prediction, sms], drift_type)

    return jsonify({
        "result": prediction,
        "classifier": model_name,
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
    _, model_name = load_best_clf()

    return jsonify({
        "result": "Spam",
        "classifier": model_name,
        "sms": sms
    })


# http://localhost:8080/metrics
@app.route('/metrics', methods=['GET'])
def metrics_dump():
    """
    Return the currently saved metrics to Prometheus
    :return: Current metrics as understandable for Prometheus.
    """
    return metricsManager.get_prometheus_metrics_string()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
