'''Download and extract data.'''
import zipfile
import urllib.request

from train_model.util import DATASET_DIR

# URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
URL = 'https://surfdrive.surf.nl/files/index.php/s/WCPP8WJPrtCbUO5/download'

def main():
    '''Extract url contents to dataset directory.'''
    zip_path, _ = urllib.request.urlretrieve(URL)
    with zipfile.ZipFile(zip_path, "r") as file:
        file.extractall(DATASET_DIR)

if __name__ == 'main':
    main()
