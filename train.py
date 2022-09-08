import argparse
import pickle
from typing import Text
import numpy as np
import random
from gensim.models.word2vec import Word2Vec, PathLineSentences
import sys
import os
import string

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer


def make_pairs(corpus):
    for i in range(len(corpus) - 1):
        yield (corpus[i], corpus[i + 1])


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

class TextGen:
    def __init__(self) -> None:
        pass


    def fit(self, data_folder, model_filename, dictionary_filename):
        file_lst = os.listdir(data_folder)
        corpus = ""
        for file_name in file_lst:
            text = open(os.path.join(data_folder, file_name), encoding = "utf-8").read().lower()
            str_punct = string.punctuation
            str_punct += '—' + '»' + '«'
            str_punct = str_punct.replace('-', '')
            for p in str_punct:
                if p in text:
                    text = text.replace(p, '')
                if '“' in text:
                    text = text.replace(p, '')
            corpus += text

        corpus = text.split()
        pairs = make_pairs(corpus)

        word_dict = {}

        for word_1, word_2 in pairs:
            if word_1 in word_dict.keys():
                word_dict[word_1].append(word_2)
            else:
                word_dict[word_1] = [word_2]
        save_obj(word_dict, dictionary_filename)
        lst = list()
        lst.append(corpus)
        model = Word2Vec(lst, min_count = 1)
        model.save(model_filename)


    def generate(self, model_filename, length, prefix, dictionary_filename):
        word_dict = load_obj(dictionary_filename)
        model = Word2Vec.load(model_filename)
        chain = [prefix] 
        word = prefix.split()[-1] if prefix != "" else ""
        for i in range(int(length)):
            if word in word_dict.keys():
                weights_array = list()
                for word_candidate in set(word_dict[word]):
                    weights_array.append((word_candidate, len(word_candidate) * model.wv.similarity(word_candidate, word)))
                weights_array = sorted(weights_array, key=lambda weight: weight[1])
                if len(weights_array) > 20:
                    word = random.choice(weights_array[-20:-1])[0]
                else:
                    word = random.choice(weights_array)[0]
            else:
                word = random.choice(list(word_dict))
            chain.append(str(word))
        s = ' '.join(chain)
        print(s[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="./data/", help="This is the path to directory with text documents")
    parser.add_argument("--model", default="output1", help="This is the path to the file where model will be saved")

    args = parser.parse_args()
    gen = TextGen()
    gen.fit(args.inputdir, args.model, "dictionary.pkl")







