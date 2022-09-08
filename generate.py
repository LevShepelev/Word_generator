import argparse
import pickle
import numpy as np
import random
from gensim.models.word2vec import Word2Vec, PathLineSentences
import sys
import train as tr
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", help="word to start")
    parser.add_argument("--model",  default="output1", help="This is the path to the file with model")
    parser.add_argument("--length", default=20,  help="number of words to generate")

    args = parser.parse_args()

    gen = tr.TextGen()
    gen.generate(args.model, args.length, args.prefix, "dictionary.pkl")