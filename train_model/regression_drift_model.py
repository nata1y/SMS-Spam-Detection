import joblib
from scipy.stats import stats
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR

from regression_model.get_predictions import _load_data as _load_prediction_data
from train_model.text_preprocessing import prepare, _extract_message_len, _text_process

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import load
from deploy_model.util import load_best_clf


def train_regression_model():
    raw_data = _load_prediction_data()
    drift_detector = SVR()
    classifier, _ = load_best_clf()
    preprocessor = joblib.load('output/preprocessor.joblib')
    le = joblib.load('output/label_encoder.joblib')
    percentiles_stats = []
    scores = []

    for batch in range(100):
        print(f"Train logistic drift detector epoch {batch}")

        X_sample, _ = train_test_split(raw_data, test_size=0.3, random_state=batch)
        y_sample = X_sample['label']
        X_sample = preprocessor.transform(X_sample['message'])

        classifier_stats = [x[0] for x in classifier.predict_proba(X_sample)]
        classifier_res = classifier.predict(X_sample)
        percentiles_stats += [[np.percentile(classifier_stats, i) for i in range(0, 101, 5)]]
        scores += [accuracy_score(classifier_res, y_sample)]

    drift_detector.fit(percentiles_stats, scores)
    joblib.dump(drift_detector, 'regression_output/regression_model.joblib')


if __name__ == "__main__":
    train_regression_model()
