import json
import requests
import pandas as pd

from deploy_model.util import ensure_path_exists, progressBar
from train_model.text_preprocessing import _load_data as _load_training_data
from production_endpoint.generate_drifts import generate_all_drifts

drift_directory = 'dataset/drifts_incoming/'
ensure_path_exists(drift_directory)
ensure_path_exists('dataset/regression')

class DriftTypes():

    def __init__(self) -> None:
        generate_all_drifts()

    drifts = dict(
        RANDOM = 'drift_random_0.5.txt',
        FLIP = 'drift_random_0.5.txt',
        MUTATION = 'drift_mutation.txt',
        CONCEPT = 'drift_concept.txt',
        HAM = 'drift_ham_only.txt',
        SPAM = 'drift_spam_only.txt'
    )
    current_name = 'RANDOM'

    def get_drifts(self):
        return self.drifts

    def get_current_name(self):
        return self.current_name

    def get_current_filename(self):
        return self.drifts[self.current_name]

    def set_current_name(self, name):
        self.current_name = name

def _load_drift(filename):
    messages = pd.read_csv(drift_directory + filename,
        sep='\t', names=['label', 'message', 'real_label'])
    return messages


def _load_data():
    messages = pd.read_csv(
        'dataset/regression/SMSSpamCollection_diff',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def main():
    vanilla_train = _load_training_data()

    print("Starting vanilla train run")
    for i, row in vanilla_train.iterrows():
        progressBar(i, len(vanilla_train))
        res = requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
                            json={'sms': row['message'], 'real_label': row['label'], drift_type': 'VANILLA_TRAINING'})

    vanilla_test = _load_data()

    print("Starting vanilla test run")
    for i, row in vanilla_test.iterrows():
        progressBar(i, len(vanilla_test))
        res = requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
                            json={'sms': row['message'], 'real_label': row['label'], 'drift_type': 'VANILLA_INCOMING'})

    dt = DriftTypes()
    drifts = dt.get_drifts()
    for key in drifts:
        print("Starting drift " + key + "\n")
        dt.set_current_name(key)
        raw_drift = _load_drift(dt.get_current_filename())

        for i, row in raw_drift.iterrows():
            progressBar(i, len(raw_drift))
            res = requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
                                json={'sms': row['message'], 'label': row['label'], 'real_label': row['real_label'], 'drift_type': dt.get_current_name(), 'window_size': round(len(raw_drift) / 6)})


if __name__ == "__main__":
    main()
