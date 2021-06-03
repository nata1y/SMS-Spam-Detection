import pandas as pd
import numpy as np

from alibi_detect.cd import KSDrift, ClassifierUncertaintyDrift, MMDDrift, ChiSquareDrift

from regression_model.get_predictions import _load_data as _load_prediction_data
from train_model.text_preprocessing import _load_data, _preprocess, _label_encoder
from train_model.generate_drifts import create_random_drift

def _load_random_drift():
    print("Loading random drift...")
    create_random_drift()
    messages = pd.read_csv(
        'dataset/drifts/drift_random.txt',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def _get_drifts(values, p_val):
    ks_drift = KSDrift(values, p_val=p_val)
    # cu_drift = ClassifierUncertaintyDrift(values, p_val=p_val) # requires model
    cs_drift = ChiSquareDrift(values, p_val=p_val)

    return ks_drift, cs_drift

def _get_num_drifts(num_values, p_val):
    mmd_drift = MMDDrift(num_values, p_val=p_val) # requires numerical values
    return mmd_drift

def _detect_drift(data, preprocessed=[]):
    raw_real_data = _load_data()
    p_val = 0.05
    print(raw_real_data.values)
    cds = _get_drifts(raw_real_data.values, p_val)

    for cd in cds:
        preds = cd.predict(data.values)
        print(preds['meta']['name'] + ": " + str(preds['data']))

    if len(preprocessed) > 0:
        real_preprocessed = _label_encoder()
        real_preprocessed = np.reshape(real_preprocessed, (len(real_preprocessed), 1))
        cd = _get_num_drifts(real_preprocessed, p_val)
        preprocessed = np.reshape(preprocessed, (len(preprocessed), 1))
        preds = cd.predict(preprocessed)
        print(preds['meta']['name'] + ": " + str(preds['data']))

def main():
    raw_drift_data = _load_random_drift()
    _detect_drift(raw_drift_data)
    

if __name__ == "__main__":
    main()
