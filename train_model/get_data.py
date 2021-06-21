'''Download and extract training data.'''
import zipfile
import urllib.request

from train_model.util import DATASET_DIR
from deploy_model.util import ensure_path_exists

# URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
URL = 'https://surfdrive.surf.nl/files/index.php/s/WCPP8WJPrtCbUO5/download'

ensure_path_exists(DATASET_DIR)

zip_path, _ = urllib.request.urlretrieve(URL)
with zipfile.ZipFile(zip_path, "r") as file:
    file.extractall(DATASET_DIR)
