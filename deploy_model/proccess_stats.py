import json
import random
from datetime import datetime

from joblib import load
from sklearn.model_selection import cross_val_score

from deploy_model.util import ensure_path_exists
from train_model.nlp_sms_model import doc_distance
import pandas as pd
from ot.lp import wasserstein_1d

ensure_path_exists('output/stats')
amount_subsamples = 100

try:
    stats_nlp = pd.read_csv('output/stats/nlp_stats.csv')
except Exception as e:
    print(e)
    # TODO: add pass to dumped data
    stats_nlp = pd.DataFrame([], columns=["date", "kl_divergence"])
    stats_nlp.to_csv('output/stats/nlp_stats.csv')


try:
    stats_loss = pd.read_csv('output/stats/loss_stats.csv')
except Exception as e:
    print(e)
    # TODO: add pass to dumped data
    stats_loss = pd.DataFrame([], columns=["date", "loss_dist"])
    stats_loss.to_csv('output/stats/loss_stats.csv')


def compare_loss_dist(doc, model, y=[]):
    global stats_loss
    now = datetime.now()
    le = load('output/label_encoder.joblib')

    # balanced_accuracy, log_loss
    losses = cross_val_score(model, doc['message'], le.transform(y), cv=amount_subsamples, scoring='neg_brier_score')
    train_losses = json.loads('output/losses.json')['losses']
    loss_dist = wasserstein_1d(losses, train_losses)
    stats_loss = stats_loss.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "loss_dist": loss_dist
                }, ignore_index=True)
    stats_loss.to_csv('output/stats/loss_stats.csv')


def compare_nlp_models(doc):
    global stats_nlp
    now = datetime.now()
    distance = doc_distance(doc)
    stats_nlp = stats_nlp.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "kl_divergence": distance
                }, ignore_index=True)
    stats_nlp.to_csv('output/stats/nlp_stats.csv')
