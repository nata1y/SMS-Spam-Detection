import pandas as pd

from alibi_detect.cd import KSDrift, ClassifierUncertaintyDrift, MMDDrift, ChiSquareDrift

from regression_model.get_predictions import _load_data as _load_prediction_data
from train_model.text_preprocessing import _load_data, _preprocess
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
    # mmd_drift = MMDDrift(values, p_val=p_val) # requires numerical values
    cs_drift = ChiSquareDrift(values, p_val=p_val)

    return ks_drift, cs_drift

def main():
    raw_real_data = _load_data()
    raw_drift_data = _load_random_drift()

    cds = _get_drifts(raw_real_data.values, 0.05)

    for cd in cds:
        preds = cd.predict(raw_drift_data.values)
        print(preds['meta']['name'] + ": " + str(preds['data']))

if __name__ == "__main__":
    main()
