import pandas as pd

from alibi_detect.cd import KSDrift

from regression_model.get_predictions import _load_data as _load_prediction_data
from train_model.text_preprocessing import _load_data, _preprocess
from train_model.generate_drifts import create_random_drift

def _load_drift():
    create_random_drift()
    messages = pd.read_csv(
        'dataset/drifts/drift_random.txt',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def main():
    raw_real_data = _load_data()

    # cd = KSDrift(raw_real_data.values, preprocess_fn=_preprocess)
    cd = KSDrift(raw_real_data.values, p_val=0.05)

    preds = cd.predict(raw_real_data.values)
    print(preds)

    raw_drift_data = _load_drift()

    preds = cd.predict(raw_drift_data.values)
    print(preds)


if __name__ == "__main__":
    main()