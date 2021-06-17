'''Utility functions for the deploy module.'''
import os
from datetime import datetime
from pathlib import Path
from joblib import load


def progress_bar(current, total, bar_length = 20):
    '''Visual progress bar in console.'''
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def ensure_path_exists(path):
    '''Creates directory if not exists.'''
    Path(path).mkdir(parents=True, exist_ok=True)

def remove_file(path):
    '''Remove directory if exists.'''
    Path(path).unlink(missing_ok=True)

# pylint: disable=W0703
def load_best_clf():
    '''Loads the best original model.'''
    file_to_load, latest_date = 'model.joblib', datetime.strptime('01-01-1970', "%m-%d-%Y")
    for filename in os.listdir('output'):
        if filename.endswith(".joblib"):
            try:
                if latest_date < datetime.strptime(
                        filename.split('_')[1].split('.')[0], "%m-%d-%Y"):
                    file_to_load = filename
                    latest_date = datetime.strptime(
                        filename.split('_')[1].split('.')[0], "%m-%d-%Y")
                    classifier_name = filename.split('_')[0]
            except Exception:
                continue

    return load('output/' + file_to_load), classifier_name
