from pathlib import Path


def ensure_path_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def remove_file(path):
    Path(path).unlink(missing_ok=True)
