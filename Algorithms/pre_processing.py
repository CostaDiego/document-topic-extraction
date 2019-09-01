import pandas as pd
import gensim


def import_files (path, verbose = True):
    data = pd.read_csv(path)
    
    if verbose is True:
        print("Input Format:")
        print('Rows: {}, Columns: {}\n'.format(data.shape[0], data.shape[1]))

    data = data.dropna().reset_index(drop=True)
    
    if verbose is True:
        print("Pre cleaning format:")
        print('Rows: {}, Columns: {}'.format(data.shape[0], data.shape[1]))

    return data

    