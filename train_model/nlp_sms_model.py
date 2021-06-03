from datetime import datetime

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mutual_info_score as kl_divergence
from sklearn.metrics.pairwise import linear_kernel
from ot.lp import wasserstein_1d

from deploy_model.util import ensure_path_exists
import numpy as np

ensure_path_exists('output/nlp_drift')


def doc_distance(doc):
    gold_standard = load('output/nlp_drift/train_data_tfidf.joblib')
    nlp_model = load('output/nlp_drift/nlp_model.joblib')
    gold_standard_features = nlp_model.get_feature_names()
    # print(nlp_model.transform(doc).todense().shape)
    # print(gold_standard.todense().shape)
    # print(gold_standard.toarray())

    # cosine_similarities = linear_kernel(gold_standard.todense(), nlp_model.transform(doc).todense()).flatten()
    # similarity = sum(cosine_similarities)

    res = wasserstein_1d(np.sum(gold_standard.toarray(), axis=0), np.sum(nlp_model.transform(doc).toarray(), axis=0))
    # print(res)

    return res


def train_nlp_model(messages):
    nlp_model = CountVectorizer()
    gold_standard = nlp_model.fit_transform(messages['message'])
    dump(nlp_model, 'output/nlp_drift/nlp_model.joblib')
    dump(gold_standard, 'output/nlp_drift/train_data_tfidf.joblib')
