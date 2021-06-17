'''NLP Model trained on the initial training data.'''
import numpy as np

from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from ot.lp import wasserstein_1d

from deploy_model.util import ensure_path_exists
from train_model.util import OUTPUT_NLP_DIR

class NLPModel():
    '''Class containing the NLP Model training methods.'''

    model: any
    gold_standard: any
    data_path: str
    model_path: str

    def __init__(self) -> None:
        self.model = CountVectorizer()
        self.data_path = OUTPUT_NLP_DIR + 'train_data_tfidf.joblib'
        self.model_path = OUTPUT_NLP_DIR + 'nlp_model.joblib'
        ensure_path_exists('output/nlp_drift')

    def load_model(self):
        '''Loads the model and gold standard data for predictions.'''
        self.gold_standard = load(self.data_path)
        self.model = load(self.model_path)

    def doc_distance(self, doc):
        '''Calculates the document distance.'''
        self.load_model()
        return wasserstein_1d(np.sum(self.gold_standard.toarray(), axis=0),
            np.sum(self.model.transform(doc).toarray(), axis=0))

    def train_nlp_model(self, messages):
        '''Trains the model.'''
        gold_standard = self.model.fit_transform(messages['message'])
        dump(self.model, self.data_path)
        dump(gold_standard, self.model_path)
