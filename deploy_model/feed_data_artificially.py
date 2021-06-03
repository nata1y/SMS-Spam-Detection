import copy
import pandas as pd

from deploy_model.proccess_stats import compare_nlp_models, compare_loss_dist
from deploy_model.util import progressBar

def get_loss_and_nlp(data, predictions, stats, types=["loss", "nlp"]):
    dt = 'train_data'
    losses = []
    stats_nlp, stats_loss = None, None

    for idx, row in predictions.iterrows():
        progressBar(idx, len(predictions))
        losses.append(0.0 if data.loc[idx, 'label'] == row['label'] else 1.0)

    if (stats.shape[0]) % 100 == 0:
        if "nlp" in types:
            stats_nlp = compare_nlp_models(stats["sms"].tolist()[-100:], dt)
        if "loss" in types:
            stats_loss = compare_loss_dist(losses, dt)
        losses = []

    return stats_nlp, stats_loss

def main():
    data = pd.read_csv(
        'dataset/SMSSpamCollection',
        sep='\t',
        names=['label', 'message']
    )
    get_loss_and_nlp(data)

if __name__ == '__main__':
    main()
