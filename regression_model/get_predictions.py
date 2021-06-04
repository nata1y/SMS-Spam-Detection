import json

import pandas as pd
import numpy as np
import requests

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from deploy_model.util import ensure_path_exists, progressBar

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


def main():
    raw_data = _load_data()

    for i, row in raw_data.iterrows():
        progressBar(i, len(raw_data))
        res = requests.post("http://127.0.0.1:8080/predict", headers={'Content-Type': 'application/json'},
                            json={'sms': row['message']})
        try:
            data = json.loads(res.content.decode('utf-8'))
            # print(f"RESPONSE: {data}")
        except:
            print(res.content)


if __name__ == "__main__":
    main()
