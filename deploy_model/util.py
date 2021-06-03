from pathlib import Path


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def ensure_path_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def remove_file(path):
    Path(path).unlink(missing_ok=True)
