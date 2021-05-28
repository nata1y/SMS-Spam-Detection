from datetime import datetime

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score as kl_divergence

from deploy_model.util import ensure_path_exists

ensure_path_exists('output/nlp_drift')


def doc_distance(doc):
    gold_standard = load('output/nlp_drift/train_data_tfidf.joblib')
    nlp_model = load('output/nlp_drift/nlp_model.joblib')
    gold_standard_features = nlp_model.get_feature_names()
    return kl_divergence(gold_standard.data, nlp_model.transform(doc).data)


def train_nlp_model(messages):
    nlp_model = TfidfVectorizer()
    gold_standard = nlp_model.fit_transform(messages['message'])
    dump(nlp_model, 'output/nlp_drift/nlp_model.joblib')
    dump(gold_standard, 'output/nlp_drift/train_data_tfidf.joblib')
