import pandas as pd
import numpy as np
from joblib import load

from deploy_model.proccess_stats import compare_nlp_models, compare_loss_dist, get_regression_predictions
from deploy_model.util import progressBar



def get_all_stats(data, predictions, clf, window, types=["loss", "nlp", "reg"]):
    dt = 'api'
    losses = []
    stats_nlp, stats_loss, regression_stats = None, None, None
    preprocessor = load('output/preprocessor.joblib')

    for idx, row in predictions.iterrows():
        progressBar(idx, len(predictions))
        losses.append(0.0 if data[idx] == row['label'] else 1.0)

    classifier_stats = [x[0] for x in clf.predict_proba(preprocessor.transform(predictions['message']))]
    percentile = [[np.percentile(classifier_stats, i) for i in range(0, window, 5)]]

    if (predictions.shape[0]) % window == 0:
        if "nlp" in types:
            stats_nlp = compare_nlp_models(predictions["message"].tolist()[-window:], dt)
        if "loss" in types:
            stats_loss = compare_loss_dist(losses[-window:], dt)
        if "reg" in types:
            regression_stats = get_regression_predictions(percentile, dt)

    return stats_nlp, stats_loss, regression_stats


def main():
    data = pd.read_csv(
        'dataset/SMSSpamCollection',
        sep='\t',
        names=['label', 'message']
    )
    # get_all_stats(data)


if __name__ == '__main__':
    main()
