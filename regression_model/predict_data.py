from matplotlib.pyplot import axis
import pandas as pd
import numpy as np

import get_predictions
from train_model.text_preprocessing import _load_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import load

def _get_predictions():
    predictions = pd.read_csv(
        "regression_dataset/predictions",
        sep="\t",
        names=['message', 'result']
    )
    return predictions

def my_train_test_split(*datasets):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    '''
    return train_test_split(*datasets, test_size=0.3, random_state=101)

def predict_labels(classifier, X_test):
    return classifier.predict(X_test)

def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)

def main():
    raw_data = get_predictions._load_data()
    df = pd.DataFrame(_get_predictions(), columns=['message', 'result'])
    preprocessed_data = load('regression_output/preprocessed_data.joblib')

    (X_train, X_test,
     y_train, y_test,
     _, test_messages) = my_train_test_split(preprocessed_data,
                                             raw_data['label'],
                                             raw_data['message'])

    classifier = LogisticRegression()
    train_classifier(classifier, X_train, y_train)
    pred = predict_labels(classifier, X_test)
    pred_scores = [accuracy_score(y_test, pred)]

    print('####### Actual Score ########')
    print(classification_report(y_test, pred))
    print(pred_scores)

    (X_train, X_test,
     y_train, y_test,
     _, test_messages) = my_train_test_split(preprocessed_data,
                                             df['result'],
                                             raw_data['message'])

    classifier = LogisticRegression()
    train_classifier(classifier, X_train, y_train)
    pred = predict_labels(classifier, X_test)
    pred_scores = [accuracy_score(y_test, pred)]

    print('####### Prediction Score ########')
    print(classification_report(y_test, pred))
    print(pred_scores)

    raw_data = _load_data()
    preprocessed_data = load('output/preprocessed_data.joblib')

    (X_train, X_test,
     y_train, y_test,
     _, test_messages) = my_train_test_split(preprocessed_data,
                                             raw_data['label'],
                                             raw_data['message'])

    classifier = LogisticRegression()
    train_classifier(classifier, X_train, y_train)
    pred = predict_labels(classifier, X_test)
    pred_scores = [accuracy_score(y_test, pred)]

    print('####### Model Score ########')
    print(classification_report(y_test, pred))
    print(pred_scores)

if __name__ == "__main__":
    main()