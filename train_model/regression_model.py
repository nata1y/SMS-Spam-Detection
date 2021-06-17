'''Regression model trained on different types of drift accuracy on original model.'''
import numpy as np

from joblib import load, dump
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from train_model.text_preprocessing import prepare, _extract_message_len, _text_process
from deploy_model.util import load_best_clf
from train_model.util import load_data, DATASET_DIR, DATA_DRIFT_DIR

class RegressionModel():
    '''Class containing the Regression Model training methods.'''
    datasets: list
    drift_detector: any
    classifier: any
    preprocessor: any

    def __init__(self) -> None:
        self.set_datasets()
        self.drift_detector = SVR()
        self.classifier, _ = load_best_clf()
        self.preprocessor = load('output/preprocessor.joblib')

    def set_datasets(self):
        '''Set the datasets to train model on.'''
        self.datasets = [DATASET_DIR + 'SMSSpamCollection',
                        DATA_DRIFT_DIR + 'drift_flip.txt',
                        DATA_DRIFT_DIR + 'drift_random_0.5.txt',
                        DATA_DRIFT_DIR + 'drift_mutation.txt',
                        DATA_DRIFT_DIR + 'drift_concept.txt',
                        DATA_DRIFT_DIR + 'drift_ham_only.txt',
                        DATA_DRIFT_DIR + 'drift_spam_only.txt']

    def train_regression_model(self):
        '''Trains the regression model on all supplied datasets.'''
        percentiles_stats = []
        scores = []

        for index, data_set in enumerate(self.datasets):
            raw_data = load_data(data_set)
            for batch in range(25):
                print(f"Train logistic drift detector epoch {batch}, dataset {index}")

                x_sample, _ = train_test_split(raw_data, test_size=0.3, random_state=batch)
                y_sample = x_sample['label']
                x_sample = self.preprocessor.transform(x_sample['message'])

                classifier_stats = [x[0] for x in self.classifier.predict_proba(x_sample)]
                classifier_res = self.classifier.predict(x_sample)
                print(accuracy_score(classifier_res, y_sample))

                percentiles_stats += [
                    [np.percentile(classifier_stats, i) for i in range(0, 101, 5)]]
                scores += [accuracy_score(classifier_res, y_sample)]

        self.drift_detector.fit(percentiles_stats, scores)
        dump(self.drift_detector, 'output/regression/regression_model.joblib')


if __name__ == "__main__":
    RegressionModel().train_regression_model()
