import joblib
from scipy.stats import stats
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR

from train_model.text_preprocessing import prepare, _extract_message_len, _text_process

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import load
from deploy_model.util import load_best_clf
import pandas as pd


datasets = ['regression_dataset/SMSSpamCollection_diff',
            'dataset/drifts/drift_random_0.5.txt',
            'dataset/drifts/drift_mutation.txt']


def _load_data(set):
    messages = pd.read_csv(
        set,
        sep='\t',
        names=['label', 'message']
    )
    return messages


def train_regression_model():
    global datasets
    drift_detector = SVR()
    classifier, _ = load_best_clf()
    preprocessor = joblib.load('output/preprocessor.joblib')
    le = joblib.load('output/label_encoder.joblib')
    percentiles_stats = []
    scores = []

    for data_set in datasets:
        raw_data = _load_data(data_set)
        for batch in range(100):
            print(f"Train logistic drift detector epoch {batch}")

            X_sample, _ = train_test_split(raw_data, test_size=0.3, random_state=batch)
            y_sample = X_sample['label']
            X_sample = preprocessor.transform(X_sample['message'])

            classifier_stats = [x[0] for x in classifier.predict_proba(X_sample)]
            classifier_res = classifier.predict(X_sample)
            percentiles_stats += [[np.percentile(classifier_stats, i) for i in range(0, 101, 5)]]
            print(accuracy_score(classifier_res, y_sample))
            scores += [accuracy_score(classifier_res, y_sample)]

    drift_detector.fit(percentiles_stats, scores)
    joblib.dump(drift_detector, 'regression_output/regression_model.joblib')


if __name__ == "__main__":
    train_regression_model()
