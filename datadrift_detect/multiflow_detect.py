import numpy as np
from skmultiflow.drift_detection.eddm import EDDM

from train_model.text_preprocessing import _load_data
from regression_model.predict_data import _load_prediction_data

def _use_EDDM_detect():
    eddm = EDDM()
    raw_real_data = _load_data()
    raw_pred_data = _load_prediction_data()

    for row in raw_real_data.iterrows():
        if row[1]['label'] == 'ham':
            eddm.add_element(1)
        elif row[1]['label'] == 'spam':
            eddm.add_element(0)
    
    for row in raw_pred_data.iterrows():
        if row[1]['label'] == 'ham':
            eddm.add_element(1)
        elif row[1]['label'] == 'spam':
            eddm.add_element(0)
        if eddm.detected_warning_zone():
            print("Warning zone detected @ " + str(row[0]))
        if eddm.detected_change():
            print("Change detected")
    
    print(eddm.get_info())

def main():
    _use_EDDM_detect()

if __name__ == "__main__":
    main()