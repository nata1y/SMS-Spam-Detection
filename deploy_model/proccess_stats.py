'''Methods that compare the different models.'''
import json
import random
from datetime import datetime
import pandas as pd

from joblib import load
from sklearn.metrics import mutual_info_score as kl_divergence

from deploy_model.util import ensure_path_exists
from train_model.nlp_model import NLPModel

ensure_path_exists('output/stats')
SUB_AMOUNT = 100
SAMPLE_AMOUNT = 80


def get_losses(losses, amount_subsamples):
    '''Get the subsample of the loss distribution.'''
    res = []
    for _ in range(amount_subsamples):
        loss_samples = random.sample(losses, SAMPLE_AMOUNT)
        res.append(sum(loss_samples) / len(loss_samples))

    return res

def compare_loss_dist(losses_curr, drift_type):
    '''Get the predictions of the loss distribution model.'''
    try:
        stats_loss = pd.read_csv('output/stats/loss_stats.csv')
    except FileNotFoundError:
        stats_loss = pd.DataFrame([], columns=["date", "loss_dist", "drift_type"])
        stats_loss.to_csv('output/stats/loss_stats.csv', index=False)

    now = datetime.now()

    losses = get_losses(losses_curr, SUB_AMOUNT)

    with open('output/losses.json', 'r') as j:
        train_losses = json.loads(j.read())['losses']

    loss_dist = kl_divergence(losses, train_losses)
    stats_loss = stats_loss.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "loss_dist": loss_dist,
                    "drift_type": drift_type
                }, ignore_index=True)
    stats_loss.to_csv('output/stats/loss_stats.csv', index=False)
    return stats_loss

def compare_nlp_models(doc, drift_type):
    '''Get the predictions of the NLP model.'''
    try:
        stats_nlp = pd.read_csv('output/stats/nlp_stats.csv')
    except FileNotFoundError:
        stats_nlp = pd.DataFrame([], columns=["date", "kl_divergence", "drift_type"])
        stats_nlp.to_csv('output/stats/nlp_stats.csv', index=False)

    now = datetime.now()
    distance = NLPModel().doc_distance(doc)
    stats_nlp = stats_nlp.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "kl_divergence": distance,
                    "drift_type": drift_type
                }, ignore_index=True)
    stats_nlp.to_csv('output/stats/nlp_stats.csv', index=False)
    return stats_nlp

def get_regression_predictions(percentiles, drift_type):
    '''Get the predictions of the regression model.'''
    try:
        stats_regression = pd.read_csv('output/stats/regression_stats.csv')
    except FileNotFoundError:
        stats_regression = pd.DataFrame([], columns=["date", "predicted_performance", "drift_type"])
        stats_regression.to_csv('output/stats/regression_stats.csv', index=False)

    now = datetime.now()
    reg_model = load('output/regression/regression_model.joblib')
    res = min(1.0, max(0.0, reg_model.predict(percentiles)[0]))
    stats_regression = stats_regression.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "predicted_performance": res,
                    "drift_type": drift_type
                }, ignore_index=True)
    stats_regression.to_csv('output/stats/regression_stats.csv', index=False)
    return stats_regression
