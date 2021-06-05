import json
import random
from datetime import datetime

from joblib import load
from sklearn.model_selection import cross_val_score, cross_validate

from deploy_model.util import ensure_path_exists
from train_model.nlp_sms_model import doc_distance
import pandas as pd
from ot.lp import wasserstein_1d
import numpy as np
from sklearn.metrics import mutual_info_score as kl_divergence

from train_model.text_preprocessing import prepare

ensure_path_exists('output/stats')
amount_subsamples = 100


def get_losses(losses, amount_subsamples):
    res = []
    for iter in range(amount_subsamples):
        loss_samples = random.sample(losses, 80)
        res.append(sum(loss_samples) / len(loss_samples))

    return res


def compare_loss_dist(losses_curr, dt):
    try:
        stats_loss = pd.read_csv('output/stats/loss_stats.csv')
    except Exception as e:
        stats_loss = pd.DataFrame([], columns=["date", "loss_dist", "drift_type"])
        stats_loss.to_csv('output/stats/loss_stats.csv', index=False)

    now = datetime.now()
    le = load('output/label_encoder.joblib')

    # balanced_accuracy, log_loss

    losses = get_losses(losses_curr, amount_subsamples)

    with open('output/losses.json', 'r') as j:
        train_losses = json.loads(j.read())['losses']

    loss_dist = kl_divergence(losses, train_losses)
    stats_loss = stats_loss.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "loss_dist": loss_dist,
                    "drift_type": dt
                }, ignore_index=True)
    stats_loss.to_csv('output/stats/loss_stats.csv', index=False)
    return stats_loss


def compare_nlp_models(doc, dt):
    try:
        stats_nlp = pd.read_csv('output/stats/nlp_stats.csv')
    except Exception as e:
        stats_nlp = pd.DataFrame([], columns=["date", "kl_divergence", "drift_type"])
        stats_nlp.to_csv('output/stats/nlp_stats.csv', index=False)

    now = datetime.now()
    distance = doc_distance(doc)
    stats_nlp = stats_nlp.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "kl_divergence": distance,
                    "drift_type": dt
                }, ignore_index=True)
    stats_nlp.to_csv('output/stats/nlp_stats.csv', index=False)
    return stats_nlp


def get_regression_predictions(percentiles, dt):
    try:
        stats_regression = pd.read_csv('output/stats/regression_stats.csv')
    except Exception as e:
        stats_regression = pd.DataFrame([], columns=["date", "predicted_performance", "drift_type"])
        stats_regression.to_csv('output/stats/regression_stats.csv', index=False)

    now = datetime.now()
    reg_model = load('output/regression/regression_model.joblib')
    res = reg_model.predict(percentiles)
    stats_regression = stats_regression.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "predicted_performance": res[0],
                    "drift_type": dt
                }, ignore_index=True)
    stats_regression.to_csv('output/stats/regression_stats.csv', index=False)
    return stats_regression