import pandas as pd
import numpy as np
import json

from monitoring.MetricsManager import MetricsManager
from datadrift_detect.detect_alibi import _detect_drift
from deploy_model.feed_data_artificially import get_all_stats
from deploy_model.util import load_best_clf


class DriftManager:
    # 100 is minimum
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

    def calculate_drifts(self):
        analysis_csv_row = f"{self.drift_type},"
        print("Checking complete incoming dataset for data drift...")

        full_set = pd.DataFrame(np.array(self.data[-self.window_size:]), columns=['label', 'message'])
        for detection in _detect_drift(full_set, self.preprocessed[-self.window_size:]):
            name = detection['meta']['name']
            drift_bool = detection['data']['is_drift']
            analysis_csv_row += f"{drift_bool},"
            dist = detection['data']['distance']
            if type(dist) is np.ndarray:
                for i, v in enumerate(dist):
                    analysis_csv_row += f"{v},"
                    self.metricsManager.updateMetric("driftdetection_{}_result_{}".format(name, i), v, v)
            else:
                analysis_csv_row += f"{dist},"
                self.metricsManager.updateMetric("driftdetection_{}_result".format(name), dist, dist)

        nlp_stats, loss_stats, regression_stats = get_all_stats(
            self.real_labels[-self.window_size:],
            full_set, self.clf, self.window_size)

        # METRIC: driftdetection_nlp_results
        nlp_results: float = nlp_stats['kl_divergence'].tolist()[-1]
        nlp_results_smooth: float = nlp_stats.ewm(com=0.5).mean()['kl_divergence'].tolist()[-1]
        self.metricsManager.updateMetric("driftdetection_nlp_results", nlp_results, nlp_results_smooth)

        is_npl_drift: bool = nlp_results > self.thresholds['nlp']
        analysis_csv_row += f"{nlp_results},{(1 if is_npl_drift else 0)},"

        # METRIC: driftdetection_loss_results
        loss_results: float = loss_stats['loss_dist'].tolist()[-1]
        loss_results_smooth: float = loss_stats.ewm(com=0.5).mean()['loss_dist'].tolist()[-1]
        self.metricsManager.updateMetric("driftdetection_loss_results", loss_results, loss_results_smooth)

        is_loss_drift: bool = loss_results > self.thresholds['loss']
        analysis_csv_row += f"{loss_results},{(1 if is_loss_drift else 0)},"

        # METRIC: driftdetection_regression_results
        regression_results: float = regression_stats['predicted_performance'].tolist()[-1]
        regression_results_smooth: float = regression_stats.ewm(com=0.5).mean()['predicted_performance'].tolist()[-1]
        self.metricsManager.updateMetric("driftdetection_regression_results", regression_results, regression_results_smooth)

        is_regression_drift = regression_results < self.thresholds['regression']
        analysis_csv_row += f"{regression_results},{(1 if is_regression_drift else 0)}"

        f = open("output/combined_stats.csv", "a")
        f.write(analysis_csv_row + "\n")
        f.close()


if __name__ == '__main__':
    DriftManager().calculate_drifts()
