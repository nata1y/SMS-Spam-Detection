import pandas as pd
import numpy as np
import json

from datadrift_detect.detect_alibi import _detect_drift
from deploy_model.feed_data_artificially import get_all_stats
from deploy_model.proccess_stats import get_regression_predictions
from deploy_model.util import load_best_clf


class DriftManager:
    window_size = 100

    calls: int
    data: list
    stats: list
    incoming_real_labels: list
    preprocessed: list

    def __init__(self) -> None:
        self.name = "Manager 1"
        self.calls = 0
        self.data = np.array([])
        self.preprocessed = np.array([])
        self.clf, _ = load_best_clf()
        self.incoming_real_labels = pd.read_csv(
            'dataset/regression/SMSSpamCollection_diff',
            sep='\t',
            names=['label', 'message']
        )
        try:
            with open('output/stats/thresholds.json', 'r') as j:
                self.thresholds = json.loads(j.read())
        except:
            self.thresholds = {
                "nlp": 0.3,
                "loss": 10.0,
                "regression": 0.8
            }
            with open('output/stats/thresholds.json', 'w') as outfile:
                json.dump(self.thresholds, outfile)

    def add_call(self, prediction):
        self.calls = self.calls + 1
        if prediction[0] == 'ham':
            self.preprocessed = np.append(self.preprocessed, [0])
        elif prediction[0] == 'spam':
            self.preprocessed = np.append(self.preprocessed, [1])
        if len(self.data) == 0:
            self.data = np.array([prediction])
        else:
            self.data = np.append(self.data, [prediction], axis=0)

            if len(self.data) % self.window_size == 0:
                self.calculate_drifts()

    def calculate_drifts(self):
        # TODO: add discounted moving average?
        # print("Checking last 10 elements for data drift...")
        # indices = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        # last_10 = pd.DataFrame(np.take(self.data, indices, axis=0), columns=['label', 'message'])
        # for detection in _detect_drift(last_10, self.preprocessed):
        #     print("DRIFT DETECTED" if detection['data']['is_drift'] == 1 else "")
        #     print(detection['meta']['name'] + ": " + str(detection['data']))

        print("Checking complete incoming dataset for data drift...")
        full_set = pd.DataFrame(np.array(self.data[-self.window_size:]), columns=['label', 'message'])
        print(full_set.shape)
        for detection in _detect_drift(full_set, self.preprocessed[-self.window_size:]):
            print("DRIFT DETECTED" if detection['data']['is_drift'] == 1 else "")
            print(detection['meta']['name'] + ": " + str(detection['data']))

        print("Check for concept drift using NLP and loss distribution")
        nlp_stats, loss_stats, regression_stats = get_all_stats(
            self.incoming_real_labels[(len(self.data)-self.window_size):len(self.data)].reset_index(),
            full_set, self.clf)
        print("NLP Results:\n " + str(nlp_stats.iloc[-1:]))

        if nlp_stats['kl_divergence'].tolist()[-1] > self.thresholds['nlp']:
            print("====================== NLP DRIFT DETECTED =============================")

        print("Loss Results:\n" + str(loss_stats.iloc[-1:]))

        if loss_stats['loss_dist'].tolist()[-1] > self.thresholds['loss']:
            print("====================== LOSS DRIFT DETECTED ============================")

        print("Regression Results:\n" + str(regression_stats.iloc[-1:]))

        if regression_stats['predicted_performance'].tolist()[-1] < self.thresholds['regression']:
            print("====================== REG DRIFT DETECTED ============================")


if __name__ == '__main__':
    DriftManager().calculate_drifts()
