import pandas as pd

def import_files (path, verbose = True):
    data = pd.read_csv(path)
    if verbose is True:
        print('Rows: {}, Columns: {}'.format(data.shape[0], data.shape[1])) 
    return data

    