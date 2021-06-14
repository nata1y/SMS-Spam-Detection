import json
import requests
import pandas as pd

from deploy_model.util import ensure_path_exists, progressBar

drift_directory = 'dataset/drifts_incoming/'
ensure_path_exists(drift_directory)
ensure_path_exists('dataset/regression')

class DriftTypes():
    drifts = dict(
        RANDOM = 'drift_random_0.5.txt',
        FLIP = 'drift_random_0.5.txt',
        MUTATION = 'drift_mutation.txt',
        CONCEPT = 'drift_concept.txt',
        HAM = 'drift_ham_only.txt',
        SPAM = 'drift_spam_only.txt'
    )
    current_name = 'RANDOM'

    def get_current_name(self):
        return self.current_name

    def get_current_filename(self):
        return self.drifts[self.current_name]

    def set_current_name(self, name):
        self.current_name = name

def _load_drift(filename):
    messages = pd.read_csv(drift_directory + filename,
        sep='\t', names=['label', 'message'])
    return messages


def _load_data():
    messages = pd.read_csv(
        'dataset/regression/SMSSpamCollection_diff',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def main():
    dt = DriftTypes()
    dt.set_current_name('FLIP')
    raw_data = _load_data()
    raw_drift = _load_drift(dt.get_current_filename())

    for i, row in raw_drift.iterrows():
        progressBar(i, len(raw_drift))
        res = requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
                            json={'sms': row['message'], 'drift_type': dt.get_current_name()})
        try:
            data = json.loads(res.content.decode('utf-8'))
            print(f"RESPONSE: {data}")
        except:
            print(res.content)


if __name__ == "__main__":
    main()
