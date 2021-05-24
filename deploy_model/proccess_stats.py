from datetime import datetime

from deploy_model.util import ensure_path_exists
from train_model.nlp_sms_model import doc_distance
import pandas as pd

ensure_path_exists('output/stats')

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


def compare_loss_dist(doc):
    pass


def compare_nlp_models(doc):
    now = datetime.now()
    distance = doc_distance(doc)
    stats = stats_nlp.append({
                    "date": now.strftime("%m-%d-%Y"),
                    "kl_divergence": distance
                }, ignore_index=True)
    stats.to_csv('output/stats/nlp_stats.csv')
