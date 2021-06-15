import pandas as pd
import numpy as np
import json

from monitoring.MetricsManager import MetricsManager
from datadrift_detect.detect_alibi import _detect_drift
from deploy_model.feed_data_artificially import get_all_stats
from deploy_model.util import load_best_clf

class DriftManager:
    window_size = 500
    metricsManager: MetricsManager

    calls: int
    data: list
    stats: list
    preprocessed: list
    drift_type: str
    real_labels: list

    def __init__(self, metricsManager: MetricsManager) -> None:
        self.metricsManager = metricsManager
        self.initializeMetrics()

        self.name = "Manager 1"
        self.calls = 0
        self.data = np.array([])
        self.preprocessed = np.array([])
        self.clf, _ = load_best_clf()
        self.drift_type = ''
        self.real_labels = []
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

    def set_window_size(self, size):
        self.window_size = size

    def add_real_label(self, real_label):
        self.real_labels.append(real_label)

    def initializeMetrics(self):
        # metric: driftdetection_nlp_results
        self.metricsManager.newMetric("driftdetection_nlp_results",
                                      "Drift Detection results of the NLP-model",
                                      0)
        # metric: driftdetection_loss_results
        self.metricsManager.newMetric("driftdetection_loss_results",
                                      "Drift Detection results of the loss model",
                                      0)
        # metric: driftdetection_regression_results
        self.metricsManager.newMetric("driftdetection_regression_results",
                                      "Drift Detection results of the regression model",
                                      0)
        self.metricsManager.newMetric("driftdetection_test_metric",
                                      "Drift Detection test metric",
                                      0)

    def add_call(self, prediction, drift_type):
        self.drift_type = drift_type
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
        self.metricsManager.updateMetric("driftdetection_test_metric", self.driftdetect_test_metric)

        analysis_csv_row = f"{self.drift_type},"

        print("Checking complete incoming dataset for data drift...")
        full_set = pd.DataFrame(np.array(self.data[-self.window_size:]), columns=['label', 'message'])
        for detection in _detect_drift(full_set, self.preprocessed[-self.window_size:]):
            name = detection['meta']['name']
            drift_bool = detection['data']['is_drift']

            analysis_csv_row += f"{drift_bool},"

            dist = detection['data']['distance']
            if type(dist) is np.ndarray:
                for x in dist:
                    analysis_csv_row += f"{x}," 
            else:
                analysis_csv_row += f"{dist}," 
 



        # print("Check for concept drift using NLP and loss distribution")
        nlp_stats, loss_stats, regression_stats = get_all_stats(
            self.real_labels[-self.window_size:],
            full_set, self.clf, self.window_size)

        # METRIC: driftdetection_nlp_results
        nlp_results = str(nlp_stats.iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_nlp_results", nlp_results)
        print("NLP Results:\n " + nlp_results)

        value = nlp_stats['kl_divergence'].tolist()[-1]
        drift_bool = nlp_stats['kl_divergence'].tolist()[-1] > self.thresholds['nlp']

        analysis_csv_row += f"{value},{(1 if drift_bool else 0)},"

        # METRIC: driftdetection_loss_results
        loss_results = str(loss_stats.iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_loss_results", loss_results)
        print("Loss Results:\n" + loss_results)

        value = loss_stats['loss_dist'].tolist()[-1]
        drift_bool = loss_stats['loss_dist'].tolist()[-1] > self.thresholds['loss']

        analysis_csv_row += f"{value},{(1 if drift_bool else 0)},"

        # METRIC: driftdetection_regression_results
        regression_results = str(regression_stats.iloc[-1:])
        self.metricsManager.updateMetric("driftdetection_regression_results", regression_results)
        print("Regression Results:\n" + regression_results)

        value = regression_stats['predicted_performance'].tolist()[-1]
        drift_bool = regression_stats['predicted_performance'].tolist()[-1] < self.thresholds['regression']

        analysis_csv_row += f"{value},{(1 if drift_bool else 0)}"

        f = open("output/combined_stats.csv", "a")
        f.write(analysis_csv_row + "\n")
        f.close()


if __name__ == '__main__':
    DriftManager().calculate_drifts()
