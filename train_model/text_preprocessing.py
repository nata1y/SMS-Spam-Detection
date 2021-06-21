"""
Preprocess the data to be trained by the learning algorithm.
Creates files `preprocessor.joblib` and `preprocessed_data.joblib`
"""
import string
import nltk
import pandas as pd
import numpy as np

from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from deploy_model.util import ensure_path_exists
from train_model.util import DATASET_DIR, load_data

nltk.download('stopwords')
ensure_path_exists('output')


def _text_process(data):
    '''
    1. remove punc
    2. do stemming of words
    3. remove stop words
    4. return list of clean text words
    '''
    nopunc = [c for c in data if c not in string.punctuation] #remove punctuations
    nopunc = ''.join(nopunc)

    stemmed = ''
    nopunc = nopunc.split()
    for i in nopunc:
        stemmer = SnowballStemmer('english')
        stemmed += (stemmer.stem(i)) + ' ' # stemming of words

    clean_msgs = [
        word for word in stemmed.split()
        if word.lower() not in stopwords.words('english')
    ] # remove stopwords

    return clean_msgs


def _extract_message_len(data):
    '''Return as np.array and reshape so that it works with make_union.'''
    return np.array([len(message) for message in data]).reshape(-1, 1)


def _preprocess(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    3. Add message length
    '''
    preprocessor = make_union(
        make_pipeline(
            CountVectorizer(analyzer=_text_process),
            TfidfTransformer()
        ),
        # append the message length feature to the vector
        FunctionTransformer(_extract_message_len, validate=False)
    )

    preprocessed_data = preprocessor.fit_transform(messages['message'])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(messages['label'])
    dump(preprocessor, 'output/preprocessor.joblib')
    dump(preprocessed_data, 'output/preprocessed_data.joblib')
    dump(label_encoder, 'output/label_encoder.joblib')
    return preprocessed_data

def prepare(message):
    '''Prepare a message using the preprocessor.'''
    preprocessor = load('output/preprocessor.joblib')
    return preprocessor.transform([message])

def main():
    '''Main function preprocessing all train data.'''
    messages = load_data(DATASET_DIR + 'SMSSpamCollection')
    print('\n################### Processed Messages ###################\n')
    with pd.option_context('expand_frame_repr', False):
        print(messages)
    _preprocess(messages)


if __name__ == "__main__":
    main()
