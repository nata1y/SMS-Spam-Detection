'''Calls the model API with the incoming dataset and its drifts.'''
import requests
import pandas as pd

from deploy_model.util import ensure_path_exists, progress_bar
from train_model.util import DATASET_DIR, REGRESSION_DATA_DIR, INCOMING_DRIFT_DIR, load_data
from production_endpoint.generate_drifts import generate_all_drifts

ensure_path_exists(INCOMING_DRIFT_DIR)
ensure_path_exists('dataset/regression')

class DriftTypes():
    '''Drift Types class contains all drifts with their paths.'''

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
        '''
        Retrieve the dict of drifts.
        :return: dictionary of drift names with path.
        '''
        return self.drifts

    def get_current_name(self):
        '''
        Retrieve the current drift.
        :return: name string of current drift.
        '''
        return self.current_name

    def get_current_filename(self):
        '''
        Retrieve the current path of drift.
        :return: path string of current drift.
        '''
        return self.drifts[self.current_name]

    def set_current_name(self, name):
        '''
        Set the current drift.
        :param: name string of drift to be set.
        :return: void
        '''
        self.current_name = name

def _load_drift(filename):
    '''Load the incoming drift data.'''
    return pd.read_csv(INCOMING_DRIFT_DIR + filename,
        sep='\t', names=['label', 'message', 'real_label'])

def _load_data():
    '''Load the incoming data.'''
    return pd.read_csv(
        REGRESSION_DATA_DIR + 'SMSSpamCollection_diff',
        sep='\t', names=['label', 'message', 'real_label'])

def main():
    '''Calls the POST requests for different datasets.'''
    vanilla_train = load_data(DATASET_DIR + 'SMSSpamCollection')

    print("Starting vanilla train run")
    for i, row in vanilla_train.iterrows():
        progress_bar(i, len(vanilla_train))
        requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
            json={'sms': row['message'], 'real_label': row['label'],
            'drift_type': 'VANILLA_TRAINING'})

    vanilla_test = _load_data()

    print("Starting vanilla test run")
    for i, row in vanilla_test.iterrows():
        progress_bar(i, len(vanilla_test))
        requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
            json={'sms': row['message'], 'real_label': row['label'],
            'drift_type': 'VANILLA_INCOMING'})

    drift_types = DriftTypes()
    drifts = drift_types.get_drifts()
    for key in drifts:
        print("Starting drift " + key + "\n")
        drift_types.set_current_name(key)
        raw_drift = _load_drift(drift_types.get_current_filename())

        for i, row in raw_drift.iterrows():
            progress_bar(i, len(raw_drift))
            requests.post("http://127.0.0.1:8080/predict",
                headers={'Content-Type': 'application/json'},
                json={'sms': row['message'], 'label': row['label'], 'real_label': row['real_label'],
                'drift_type': drift_types.get_current_name(),
                'window_size': round(len(raw_drift) / 6)})


if __name__ == "__main__":
    main()
