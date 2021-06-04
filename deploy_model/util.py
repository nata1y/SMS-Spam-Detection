import os
from datetime import datetime
from pathlib import Path
from joblib import load


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def ensure_path_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_file(path):
    Path(path).unlink(missing_ok=True)


def load_best_clf():
    file_to_load, latest_date = 'model.joblib', datetime.strptime('01-01-1970', "%m-%d-%Y")
    for filename in os.listdir('output'):
        if filename.endswith(".joblib"):
            try:
                if latest_date < datetime.strptime(filename.split('_')[1].split('.')[0], "%m-%d-%Y"):
                    file_to_load = filename
                    latest_date = datetime.strptime(filename.split('_')[1].split('.')[0], "%m-%d-%Y")
                    classifier_name = filename.split('_')[0]
            except Exception as e:
                continue

    return load('output/' + file_to_load), classifier_name
