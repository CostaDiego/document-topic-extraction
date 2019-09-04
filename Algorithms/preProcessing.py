import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect
from tqdm import tqdm_notebook
from itertools import chain
from gensim.models import Phrases
from gensim import corpora, models
import numpy, time

import nltk
nltk.download('wordnet')

tqdm_notebook().pandas()

stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need',
'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'much','even', 'mr', 'bbc', 'image',
'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something',
'well', 'plus', 'really', 'actually']

def import_files (path, target, preCleaning = True, dropna = 'index', verbose = False):
    if len(path) == len(target):
        data = pd.DataFrame(columns=['articles'])
        preData = []
        for i in range(len(path)):
            if verbose is True:
                print("Loading the dataset {}.\n".format(i))
            datatmp = pd.read_csv(path[i])
            
            if verbose is True:
                print("Input Format:")
                print('Rows: {}, Columns: {}.\n'.format(datatmp.shape[0], datatmp.shape[1]))

            if preCleaning is True:
                datatmp = datatmp.dropna(dropna).reset_index(drop=True)
        
                if verbose is True:
                    print("Pre cleaning format:")
                    print('Rows: {}, Columns: {}'.format(datatmp.shape[0], datatmp.shape[1]))
            preData.append(datatmp[str(target[i])])

        if verbose is True:
            print("\nRemoving unwanted information using targets.")
        data['articles'] = pd.concat(preData, ignore_index=True)  

        return data

    else:
        print("The number of datasets does not matches the number of targets")

def language_detection(dataFile, verbose = False):
    dataFile['lang']  = dataFile.articles.progress_map(detect)
    language_count = dataFile.lang.value_counts()
    if verbose is True:
        print(language_count)
        print("\nMost Frequent language: {}".format(language_count.head(1)))
    return language_count.head(1)

def language_cleaning(dataFile, language, verbose = False):
    if verbose is True:
        print("Cleaning data using '{}' language.".format(language))
    dataFile = dataFile.loc[dataFile.lang==language]
    return dataFile

def tokenization(dataFile, level = 'word', verbose = False):
    if verbose is True:     
        print('Tokenizing data to {}\n.'.format(level))

    if level is 'word':
        dataFile['tokens_sentences'] = dataFile.articles.progress_map(sent_tokenize)
        dataFile['tokens_words'] = dataFile['tokens_sentences'].progress_map(
            lambda sentences: [word_tokenize(sentence) for sentence in sentences])

    elif level is 'sentence':
        dataFile['tokens_sentences'] = dataFile.articles.progress_map(sent_tokenize)
    
    else:
        print("Tokenization Level invalid. Please choose a valid option.")
     
    return dataFile

def POS_tagging(dataFile, verbose = False):
    if verbose is True:
        print("Applying Part Of Speech tags")

    dataFile['POS_tokens'] = dataFile['tokens_words'].progress_map(
        lambda tokens_words: [pos_tag(token) for token in tokens_words]
    )

    return dataFile

def getPOS_tagging(tagging_tree):

    if tagging_tree.startswith('J'):
        return wordnet.ADJ
    elif tagging_tree.startswith('V'):
        return wordnet.VERB
    elif tagging_tree.startswith('N'):
        return wordnet.NOUN
    elif tagging_tree.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmatizing(dataFile, verbose = False):
    lemmatizer = WordNetLemmatizer()
    
    dataFile['tokens_words_lemmatized'] = dataFile['POS_tokens'].progress_map(
    lambda list_tokens_POS: [
        [
            lemmatizer.lemmatize(el[0], getPOS_tagging(el[1])) 
            if getPOS_tagging(el[1]) != '' else el[0] for el in tokens_POS
        ] 
        for tokens_POS in list_tokens_POS
    ]
)
    return dataFile

def getStopWords(verbs = stopwords_verbs, others = stopwords_other):
    FinalStopwords = stopwords.words('english') + verbs + others

    return FinalStopwords

def removeStopWords(dataFile, minSize = 1, verbose = False):
    if verbose is True:
        print('Importing Stop words')
    
    stopwords = getStopWords()

    if verbose is True:
        print('Cleaning the tokens from {} using list of stopwords'.format(dataFile))

    dataFile['tokens'] = dataFile['tokens_words_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    dataFile['tokens'] = dataFile['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                    and token.lower() not in stopwords and len(token)>minSize])
    return dataFile

def Bi_n_TrigramModel(dataFile, min_cnt = 1, verbose = False):
    if verbose is True:
        print('Getting tokens From data file and converting into a list of tokens.')
    tokens = dataFile['tokens'].tolist()

    if verbose is True:
        print('Building the Bigram Model')
    bigram_model = Phrases(tokens)

    if verbose is True:
        print('Building the Trigram Model')
    trigram_model = Phrases(bigram_model[tokens], min_cnt)

    if verbose is True:
        print('Importing the Trigram Model and converting into list')
    tokens = list(trigram_model[bigram_model[tokens]])

    return tokens

def generateDictionary(tokens, min_thld, verbose = False):
    if verbose is True:
        print('Generating the Ditionary.')
    diction = corpora.Dictionary(tokens)

    if verbose is True:
        print('Filtering dictionary using the minimun threshold: {}'.format(min_thld))
    diction.filter_extremes(no_below = min_thld)

    return diction

def generateBOW(diction, tokens, verbose = False):
     if verbose is True:
         print('Generating Bag Of Words.')
     bow = [diction.doc2bow(token) for token in tokens]
     return bow

def trainModel(bow, dictionary, numTopics = None, numPasses = None, verbose = False):  

    if verbose is True:
        if numTopics is None:
            print('Please inform a valid number of topics.')
        if dictionary is None:
            print('Please inform a valid dictionary.')    
        if numPasses is None:
            print('Please inform a valid dictionary.')

        print('Trainning LDA model using the inputed BOW and Dictionary.')
        print('Parameters: Topics: {}, Passes: {}'.format(numTopics,numPasses))

    numpy.random.seed(int(time.time()))

    modelLDA = models.LdaModel(
        bow,
        num_topics= numTopics,
        id2word= dictionary,
        passes= numPasses,
        alpha= [0.01]*numTopics,
        eta= [0.01]*len(dictionary.keys())
    )

    return modelLDA
