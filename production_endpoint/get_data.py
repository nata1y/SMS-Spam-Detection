"""
Download and extract data.
"""
import urllib.request
import zipfile

from deploy_model.util import remove_file, ensure_path_exists
from train_model.util import REGRESSION_DATA_DIR, DATASET_DIR

# URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
# URL = 'https://surfdrive.surf.nl/files/index.php/s/OZRd9BcxhGkxTuy/download' # V2
# URL = 'https://surfdrive.surf.nl/files/index.php/s/H4e35DvjaX18pTI/download' # V3
URL = 'https://surfdrive.surf.nl/files/index.php/s/HU5mY29RzxRlHCU/download' # V4

ensure_path_exists(DATASET_DIR)
ensure_path_exists(REGRESSION_DATA_DIR)

def get_data():
    '''Get the production data.'''
    zip_path, _ = urllib.request.urlretrieve(URL)
    with zipfile.ZipFile(zip_path, "r") as file:
        file.extractall(REGRESSION_DATA_DIR)

    remove_file(REGRESSION_DATA_DIR + '/SMSSpamCollection_diff')

    with open(REGRESSION_DATA_DIR + '/SMSSpamCollection', 'r') as src, \
        open(DATASET_DIR + '/SMSSpamCollection', 'r') as diff, \
        open(REGRESSION_DATA_DIR + '/SMSSpamCollection_diff', 'a') as dest:
        nonempty_lines = [line.strip("\n") for line in diff if line != "\n"]
        ignore = len(nonempty_lines)
        i = 0
        for line in src:
            if i >= ignore:
                dest.write(line)
            i = i + 1

if __name__ == 'main':
    get_data()
