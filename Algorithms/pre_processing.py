import pandas as pd
from langdetect import detect
from tqdm import tqdm_notebook
import gensim


def import_files (path, preCleaning = True, dropna = 'index', verbose = True):
    data = pd.read_csv(path)
    
    if verbose is True:
        print("Input Format:")
        print('Rows: {}, Columns: {}\n'.format(data.shape[0], data.shape[1]))

    if preCleaning is True:
        data = data.dropna(dropna).reset_index(drop=True)
 
        if verbose is True:
            print("Pre cleaning format:")
            print('Rows: {}, Columns: {}'.format(data.shape[0], data.shape[1]))

    return data

def language_detection(dataFile, verbose = True):
    tqdm_notebook().pandas()
    dataFile['lang']  = dataFile.articles.progress_map(detect)
    language_count = dataFile.lang.value_counts()
    if verbose is True:
        print(language_count)
        print("\nMost Frequent language: {}".format(language_count.head(1)))
    return language_count.head(1)