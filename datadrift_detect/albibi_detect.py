import pandas as pd
import numpy as np

from functools import partial
from alibi_detect.cd import KSDrift, ClassifierUncertaintyDrift, MMDDrift, ChiSquareDrift
from alibi_detect.cd.tensorflow import preprocess_drift
from joblib import load

from regression_model.get_predictions import _load_data as _load_prediction_data
from train_model.text_preprocessing import _load_data, _preprocess, _text_process, _extract_message_len
from train_model.generate_drifts import create_random_drift
from deploy_model.serve_model import load_best_clf

def _load_drift():
    create_random_drift()
    messages = pd.read_csv(
        'dataset/drifts/drift_random.txt',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def _use_KS_detect():
    # Kolmogov-Smirnov 
    raw_real_data = _load_data()

    # preprocessor = load('output/preprocessor.joblib')

    # preprocess_fn = partial(preprocess_drift, model=preprocessor.fit_transform, batch_size=32)

    cd = KSDrift(raw_real_data.values, p_val=0.05)
    # cd = KSDrift(raw_real_data.values.reshape(-1, 1), p_val=0.05, preprocess_fn=preprocess_fn)

    preds = cd.predict(raw_real_data.values)
    print(preds)

    raw_drift_data = _load_drift()

    preds = cd.predict(raw_drift_data.values)
    print(preds)

def _use_CS_detect():
    # Chi Square
    raw_real_data = _load_data()

    # preprocessor = load('output/preprocessor.joblib')

    # preprocess_fn = partial(preprocess_drift, model=preprocessor.fit_transform, batch_size=32)

    cd = ChiSquareDrift(raw_real_data.values, p_val=0.05)
    # cd = KSDrift(raw_real_data.values.reshape(-1, 1), p_val=0.05, preprocess_fn=preprocess_fn)

    preds = cd.predict(raw_real_data.values)
    print(preds)

    raw_drift_data = _load_drift()

    preds = cd.predict(raw_drift_data.values)
    print(preds)

def _use_MU_detect():
    # Model Uncertainty, does NOT work expects CLF to be from Tenserflow
    raw_real_data = _load_data()
    clf = load_best_clf()

    cd = ClassifierUncertaintyDrift(raw_real_data.values, clf, p_val=0.05)

    preds = cd.predict(raw_real_data)
    print(preds)

    raw_drift_data = _load_drift()

    preds = cd.predict(raw_drift_data.values)
    print(preds)

def _use_MMD_detect():
        # Maximum Mean Descrepency, does NOT work expects number
    raw_real_data = _load_data()
    clf = load_best_clf()

    cd = MMDDrift(raw_real_data.values, p_val=0.05)

    preds = cd.predict(raw_real_data)
    print(preds)

    raw_drift_data = _load_drift()

    preds = cd.predict(raw_drift_data.values)
    print(preds)

def main():

    _use_KS_detect()
    _use_CS_detect()


    # _use_MU_detect()
    # _use_MMD_detect()
    



if __name__ == "__main__":
    main()