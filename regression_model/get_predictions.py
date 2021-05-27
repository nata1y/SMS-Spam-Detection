import pandas as pd
import numpy as np
import requests

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from deploy_model.util import ensure_path_exists

nltk.download('stopwords')
ensure_path_exists('regression_output')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from joblib import dump

def _load_data():
    
    messages = pd.read_csv(
        'regression_dataset/SMSSpamCollection_diff',
        sep='\t',
        names=['label', 'message']
    )
    return messages


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

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
    # return as np.array and reshape so that it works with make_union
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
    dump(preprocessor, 'regression_output/preprocessor.joblib')
    dump(preprocessed_data, 'regression_output/preprocessed_data.joblib')
    return preprocessed_data

def _save_predictions(predictions):
 predictions.to_csv(
     'regression_dataset/predictions',
    sep='\t',
    columns=['message', 'result'],
    index=False,
    header=False,
 )

def main():
    raw_data = _load_data()
    predictions = np.empty((0,3), str)

    for i, row in raw_data.iterrows():
        progressBar(i, len(raw_data))
        res = requests.post("http://127.17.0.2:8080/predict", headers={'Content-Type': 'application/json'}, json={'sms': row['message']})
        data = res.json()
        predictions = np.append(predictions, np.array([[row['label'], row['message'], data['result']]]), axis=0)

    df = pd.DataFrame(predictions, columns=['label', 'message', 'result'])

    _preprocess(df)
    _save_predictions(df)

if __name__ == "__main__":
    main()
