import pandas as pd
import numpy as np
import json

from monitoring.MetricsManager import MetricsManager
from datadrift_detect.detect_alibi import detect_drift
from deploy_model.feed_data_artificially import get_all_stats
from deploy_model.util import load_best_clf


class DriftManager:
    window_size = 100
    metricsManager: MetricsManager

    calls: int
    data: list
    stats: list
    incoming_real_labels: list
    preprocessed: list

    def __init__(self, metricsManager: MetricsManager) -> None:
        self.metricsManager = metricsManager
        self.initializeMetrics()

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

    def initializeMetrics(self):
        # metric: driftdetection_nlp_results
        self.metricsManager.newMetric("driftdetection_nlp_results",
                                      "Drift Detection results of the NLP-model",
                                      0, 0)
        # metric: driftdetection_loss_results
        self.metricsManager.newMetric("driftdetection_loss_results",
                                      "Drift Detection results of the loss model",
                                      0, 0)
        # metric: driftdetection_regression_results
        self.metricsManager.newMetric("driftdetection_regression_results",
                                      "Drift Detection results of the regression model",
                                      0, 0)

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

    driftdetect_test_metric: int = 0
    def calculate_drifts(self):
        self.driftdetect_test_metric += 2
        self.metricsManager.updateMetric("driftdetection_test_metric", self.driftdetect_test_metric, self.driftdetect_test_metric)

        print("Checking complete incoming dataset for data drift...")
        full_set = pd.DataFrame(np.array(self.data[-self.window_size:]), columns=['label', 'message'])
        print(full_set.shape)
        for detection in detect_drift(full_set, self.preprocessed[-self.window_size:]):
            print("DRIFT DETECTED" if detection['data']['is_drift'] == 1 else "")
            detection_data = str(detection['data'])
            print(detection['meta']['name'] + ": " + detection_data)

        print("Check for concept drift using NLP and loss distribution")
        nlp_stats, loss_stats, regression_stats = get_all_stats(
            self.incoming_real_labels[(len(self.data) - self.window_size):len(self.data)].reset_index(),
            full_set, self.clf)

        # METRIC: driftdetection_nlp_results
        nlp_results = str(nlp_stats.iloc[-1:])
        nlp_results_smooth = str(nlp_stats.ewm(com=0.5).mean().iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_nlp_results", nlp_results, nlp_results_smooth)
        print("NLP Results:\n " + nlp_results)

        if nlp_stats['kl_divergence'].tolist()[-1] > self.thresholds['nlp']:
            print("====================== NLP DRIFT DETECTED =============================")

        # METRIC: driftdetection_loss_results
        loss_results = str(loss_stats.iloc[-1:])
        loss_results_smooth = str(loss_stats.ewm(com=0.5).mean().iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_loss_results", loss_results, loss_results_smooth)
        print("Loss Results:\n" + loss_results)

        if loss_stats['loss_dist'].tolist()[-1] > self.thresholds['loss']:
            print("====================== LOSS DRIFT DETECTED ============================")

        # METRIC: driftdetection_regression_results
        regression_results = str(regression_stats.iloc[-1:])
        regression_results_smooth = str(regression_stats.ewm(com=0.5).mean().iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_regression_results", regression_results, regression_results_smooth)
        print("Regression Results:\n" + regression_results)

        if regression_stats['predicted_performance'].tolist()[-1] < self.thresholds['regression']:
            print("====================== REG DRIFT DETECTED ============================")


if __name__ == '__main__':
    DriftManager().calculate_drifts()
