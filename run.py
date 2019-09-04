import argparse
from Algorithms.modelUsageAPI import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document-topic-extraction run')
    parser.add_argument('--document', type=str, default='./Document.txt')
    parser.add_argument('--model', type=str, default='./models/LDAmodelExtended.pkl')
    parser.add_argument('--dictionary', type=str, default='./models/dictionary.dict')

    args = parser.parse_args()

    topics = infer(args.document,args.model,args.dictionary)

    print(topics)