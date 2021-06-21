# pylint: disable=R0801
'''Drift Manager manages the overall system.'''
import json
import pandas as pd
import numpy as np

from monitoring.metrics_manager import MetricsManager
from datadrift_detect.detect_alibi import detect_drift
from deploy_model.feed_data_artificially import get_all_stats
from deploy_model.util import load_best_clf


class DriftManager:
    '''Drift Manager class that handles incoming data and drift detection.'''
    metrics_manager: MetricsManager
    window_size: int
    data: list
    preprocessed: list
    drift_type: str
    real_labels: list
    thresholds: dict

    def __init__(self, metricsManager: MetricsManager) -> None:
        self.metrics_manager = metricsManager
        self.window_size = 100 # 100 is minimum
        self.data = np.array([])
        self.preprocessed = np.array([])
        self.drift_type = ''
        self.real_labels = []
        try:
            with open('output/stats/thresholds.json', 'r') as j:
                self.thresholds = json.loads(j.read())
        except FileNotFoundError:
            self.thresholds = { "nlp": 0.3, "loss": 10.0, "regression": 0.8 }
            with open('output/stats/thresholds.json', 'w') as outfile:
                json.dump(self.thresholds, outfile)

    def set_window_size(self, size):
        '''Set the window size of manager.'''
        self.window_size = size

    def add_real_label(self, real_label):
        '''Add a real label to the list.'''
        self.real_labels.append(real_label)

    def add_call(self, prediction, drift_type):
        '''Store the call to the model.'''
        self.drift_type = drift_type
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
        '''Calculate the drifts for the model.'''
        analysis_csv_row = f"{self.drift_type},"

        print("Checking complete incoming dataset for data drift...")

        full_set = pd.DataFrame(np.array(self.data[-self.window_size:]),
                    columns=['label', 'message'])

        analysis_csv_row += self.retrieve_data_results(full_set)

        clf, _ = load_best_clf()

        nlp_stats, loss_stats, regression_stats = get_all_stats(
            self.real_labels[-self.window_size:],
            full_set, clf, self.window_size)

        nlp_results: float = self.retrieve_nlp_results(nlp_stats)
        is_npl_drift: bool = nlp_results > self.thresholds['nlp']
        analysis_csv_row += f"{nlp_results},{(1 if is_npl_drift else 0)},"

        loss_results: float = self.retrieve_loss_results(loss_stats)
        is_loss_drift: bool = loss_results > self.thresholds['loss']
        analysis_csv_row += f"{loss_results},{(1 if is_loss_drift else 0)},"

        regression_results: float = self.retrieve_regression_results(regression_stats)
        is_regression_drift = regression_results < self.thresholds['regression']
        analysis_csv_row += f"{regression_results},{(1 if is_regression_drift else 0)}"

        with open("output/combined_stats.csv", "a") as file:
            file.write(analysis_csv_row + "\n")

    def retrieve_data_results(self, full_set):
        '''Retrieve data drift detection results.'''
        analysis_csv_row: str = ''
        for detection in detect_drift(full_set, self.preprocessed[-self.window_size:]):
            name = detection['meta']['name']
            drift_bool = detection['data']['is_drift']
            self.metrics_manager.update_metric(
                "driftdetection_{}_is_drift".format(name), drift_bool, drift_bool)
            analysis_csv_row += f"{drift_bool},"
            dist = detection['data']['distance']
            if isinstance(dist, np.ndarray):
                for i, value in enumerate(dist):
                    analysis_csv_row += f"{value},"
                    self.metrics_manager.update_metric(
                        "driftdetection_{}_result_{}".format(name, i), value, value)
            else:
                analysis_csv_row += f"{dist},"
                self.metrics_manager.update_metric(
                    "driftdetection_{}_result".format(name), dist, dist)
        return analysis_csv_row

    def retrieve_nlp_results(self, nlp_stats):
        '''METRIC: driftdetection_nlp_results.'''
        nlp_results: float = nlp_stats['kl_divergence'].tolist()[-1]
        nlp_results_smooth: float = nlp_stats.ewm(
            com=0.5).mean()['kl_divergence'].tolist()[-1]
        self.metrics_manager.update_metric(
            "driftdetection_nlp_results", nlp_results, nlp_results_smooth)
        return nlp_results

    def retrieve_loss_results(self, loss_stats):
        '''METRIC: driftdetection_loss_results.'''
        loss_results: float = loss_stats['loss_dist'].tolist()[-1]
        loss_results_smooth: float = loss_stats.ewm(
            com=0.5).mean()['loss_dist'].tolist()[-1]
        self.metrics_manager.update_metric(
            "driftdetection_loss_results", loss_results, loss_results_smooth)
        return loss_results

    def retrieve_regression_results(self, regression_stats):
        '''METRIC: driftdetection_regression_results.'''
        regression_results: float = regression_stats['predicted_performance'].tolist()[-1]
        regression_results_smooth: float = regression_stats.ewm(
            com=0.5).mean()['predicted_performance'].tolist()[-1]
        self.metrics_manager.update_metric(
            "driftdetection_regression_results", regression_results, regression_results_smooth)
        return regression_results

if __name__ == '__main__':
    DriftManager(MetricsManager()).calculate_drifts()
