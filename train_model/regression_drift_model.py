import pandas as pd
import numpy as np

from joblib import load, dump
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from train_model.text_preprocessing import prepare, _extract_message_len, _text_process
from deploy_model.util import load_best_clf

datasets = ['dataset/SMSSpamCollection',
            'dataset/drifts/drift_flip.txt',
            'dataset/drifts/drift_random_0.5.txt',
            'dataset/drifts/drift_mutation.txt',
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
    preprocessor = load('output/preprocessor.joblib')
    percentiles_stats = []
    scores = []

    for index, data_set in enumerate(datasets):
        raw_data = _load_data(data_set)
        for batch in range(25):
            print(f"Train logistic drift detector epoch {batch}, dataset {index}")

            X_sample, _ = train_test_split(raw_data, test_size=0.3, random_state=batch)
            y_sample = X_sample['label']
            X_sample = preprocessor.transform(X_sample['message'])

            classifier_stats = [x[0] for x in classifier.predict_proba(X_sample)]
            classifier_res = classifier.predict(X_sample)
            percentiles_stats += [[np.percentile(classifier_stats, i) for i in range(0, 101, 5)]]
            print(accuracy_score(classifier_res, y_sample))
            scores += [accuracy_score(classifier_res, y_sample)]

    drift_detector.fit(percentiles_stats, scores)
    dump(drift_detector, 'output/regression/regression_model.joblib')


if __name__ == "__main__":
    train_regression_model()
