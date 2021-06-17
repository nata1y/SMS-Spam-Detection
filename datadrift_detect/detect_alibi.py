'''Alibi detect implementation for detecting data drift with multiple algorithms.'''
import numpy as np

from alibi_detect.cd import KSDrift, MMDDrift, ChiSquareDrift #, ClassifierUncertaintyDrift

from train_model.generate_drifts import create_random_drift
from train_model.util import DATASET_DIR, DATA_DRIFT_DIR, load_data

def _label_encoder():
    '''Encodes labels to 0 or 1.'''
    labels = np.array([])
    for _, row in load_data(DATASET_DIR + 'SMSSpamCollection').iterrows():
        if row[0] == 'ham':
            labels = np.append(labels, [0], axis=0)
        elif row[0] == 'spam':
            labels = np.append(labels, [1], axis=0)
    return labels

def _get_drifts(values, p_val):
    '''Retrieve string value oriented drift algorithms.'''
    ks_drift = KSDrift(values, p_val=p_val)
    # cu_drift = ClassifierUncertaintyDrift(values, p_val=p_val) # requires model
    cs_drift = ChiSquareDrift(values, p_val=p_val)

    return ks_drift, cs_drift

def _get_num_drifts(num_values, p_val):
    '''Retrieve number value oriented drift algortihms.'''
    mmd_drift = MMDDrift(num_values, p_val=p_val) # requires numerical values
    return mmd_drift

def detect_drift(data, preprocessed=None):
    '''Detect a drift based on incoming data.'''
    raw_real_data = load_data(DATASET_DIR + 'SMSSpamCollection')
    p_val = 0.05
    cds = _get_drifts(raw_real_data.values, p_val)
    detections = np.array([])

    for drift_classifier in cds:
        preds = drift_classifier.predict(data.values)
        detections = np.append(detections, [preds])

    if preprocessed is None:
        real_preprocessed = _label_encoder()
        real_preprocessed = np.reshape(real_preprocessed, (len(real_preprocessed), 1))
        drift_classifier = _get_num_drifts(real_preprocessed, p_val)
        preprocessed = np.reshape(preprocessed, (len(preprocessed), 1))
        preds = drift_classifier.predict(preprocessed)
        detections = np.append(detections, [preds])
    return detections

def main():
    '''Main functionality executing detection on random drift.'''
    print("Loading random drift...")
    create_random_drift(0.5)
    raw_drift_data = load_data(DATA_DRIFT_DIR + 'drift_random.txt')
    detect_drift(raw_drift_data)

if __name__ == "__main__":
    main()
