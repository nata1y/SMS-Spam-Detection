'''Utilities function for train model module.'''
import pandas as pd

DATASET_DIR = 'dataset/'
DATA_DRIFT_DIR = DATASET_DIR + 'drifts/'
INCOMING_DRIFT_DIR = DATASET_DIR + 'drifts_incoming/'
REGRESSION_DATA_DIR = DATASET_DIR + 'regression/'

OUTPUT_DIR = 'output/'
OUTPUT_NLP_DIR = OUTPUT_DIR + 'nlp_drift/'

def load_data(path):
    '''Loads data from a specific path.'''
    return pd.read_csv(
        path,
        sep='\t',
        names=['label', 'message']
    )
