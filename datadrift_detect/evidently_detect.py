import pandas as pd
import numpy as np

from joblib import load
from sklearn import datasets
from evidently.dashboard import Dashboard
from evidently.tabs import DriftTab, CatTargetDriftTab, ClassificationPerformanceTab, ProbClassificationPerformanceTab

from train_model.text_preprocessing import _load_data
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
    # Does NOT work yet
    raw_real_data = _load_data()
    real_frame = pd.DataFrame(raw_real_data.values, columns=['label', 'message'])
    raw_drift_data = _load_drift()
    drift_frame = pd.DataFrame(raw_drift_data.values, columns=['label', 'message'])

    preprocessed_data = load('output/preprocessed_data.joblib')
    data = np.array([])
    for i in range(len(preprocessed_data)):
        message = np.array([])
        for j in range(len(preprocessed_data[i])):
            message = np.append(message, [preprocessed_data[i,j]])
        data = np.append(data, [message], axis=0)
    print(data)        

    pre_frame = pd.DataFrame(preprocessed_data, columns=['message'])

    iris = datasets.load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(iris_frame)
    print(drift_frame)
    print(preprocessed_data)
    print(pre_frame)

    data_and_target_drift_report = Dashboard(pre_frame[:100], pre_frame[100:], column_mapping=None, tabs=[DriftTab])
    data_and_target_drift_report.save("output/reports/my_report_with_2_tabs.html")



if __name__ == "__main__":
    main()