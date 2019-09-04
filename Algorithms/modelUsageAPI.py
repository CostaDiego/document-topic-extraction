import pickle as pkl
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim import corpora

def save(obj, path, protocol = pkl.DEFAULT_PROTOCOL, verbose = False):
    if verbose:
        print("Saving file on path: {}".format(str(path)))
    try:
        file = open(str(path), 'wb')
        pkl.dump(obj, file, protocol = pkl.DEFAULT_PROTOCOL)
        file.close()

        if verbose:
            print('Saved.')

    except pkl.PickleError:
        print('An Error occured during saving.')


def load(path, verbose = False):
    if verbose:
        print("Loading file from path: {}".format(str(path)))

    try:
        file = open(str(path), 'rb')
        obj = pkl.load(file)
        file.close()

        if verbose:
            print('Loaded.')
        
        return obj

    except pkl.PickleError:
        print('An Error occured during loading.')

def infer(document, model, dictionary, numb_topics = 12):
    print("Loading the Document from path: {}".format(str(document)))
    file = open(str(document), "r")
    doc = file.read()
    file.close()

    print("Loading model from path: {}".format(str(model)))
    mdl = load(model)

    print("Loading dictionary from path: {}".format(str(dictionary)))
    dicti = load(dictionary)

    print("Inferring:\n")
    tokens = word_tokenize(doc)
    topics = mdl.show_topics(formatted=True, num_topics = numb_topics, num_words = 20)
    infer = pd.DataFrame([(el[0], round(el[1],2), topics[el[0]][1]) 
        for el in mdl[dicti.doc2bow(tokens)]],
        columns=['topic #', 'weight', 'words in topic'])
    return infer