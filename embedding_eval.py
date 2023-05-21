from math import log
import itertools
from collections import Counter
from typing import Any
import pandas as pd
import numpy as np
from numpy.linalg import norm
from eval import calc_results
import nltk
import string
import re

from pathlib import Path
import os
import argparse
import pickle

import torch
import yaml
from gensim.models.keyedvectors import KeyedVectors
from nlp_glove_implementation.glove.src.glove import GloVe
import nlp_glove_implementation.glove.src.vocabulary as vocabulary
from nlp_glove_implementation.glove.src.vocabulary import Vocabulary
import h5py
from tqdm import tqdm

import sys
from statistics import mean, variance


# dataset: Https://Github.Com/Idontflow/Olidhttps://github.com/idontflow/OLID
DATA_DIR = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/OLID/"

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def tokenise(d):
    text_p = "".join([char for char in d.lower() if char not in string.punctuation]).replace('’', ' ')
    return nltk.word_tokenize(emoji_pattern.sub(r'', text_p))


def tweet_to_embedding(embedding, tweet, vocab=None):
    if vocab is not None:
        tweet_embedding = np.array([embedding[word] for word in tokenise(tweet) if word in vocab])
    else:
        tweet_embedding = np.array([embedding[word] for word in tokenise(tweet) if word in embedding.keys()])

    if len(tweet_embedding) != 0:
        # return tweet_embedding.mean(axis=0)
        # return tweet_embedding.sum(axis=0)
        return tweet_embedding.max(axis=0)
    else:
        return np.array([0] * 50)


def load_sgns_embedding(filepath):
    embedding = dict()
    vocab = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        vocab_size, embedding_size = lines[0].split()
        for i in range(1, len(lines)):
            line = lines[i].split()
            vocab.add(line[0])
            embedding[line[0]] = np.array([float(num) for num in line[1:]])

    return embedding, vocab


def load_config():
    # config_filepath = Path(__file__).absolute().parents[1] / "config1.yaml"
    config_filepath = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/nlp_glove_implementation/glove/config1.yaml"
    with open(config_filepath, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def load_glove_embedding(filepath):
    config = load_config()
    with open(os.path.join(config.cooccurrence_dir, "vocab.pkl"), "rb") as f:
        sys.path.append('nlp_glove_implementation/glove/src')
        vocab = pickle.load(f)

    model = GloVe(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        x_max=config.x_max,
        alpha=config.alpha
    )
    model.load_state_dict(torch.load(filepath))
    
    keyed_vectors = KeyedVectors(vector_size=config.embedding_size)
    keyed_vectors.add_vectors(
        keys=[vocab.get_token(index) for index in range(config.vocab_size)],
        weights=(model.weight.weight.detach()
            + model.weight_tilde.weight.detach()).numpy()
    )
    # print(vocab.token2index.keys())
    return keyed_vectors, vocab


def train_embeddings(embedding, vocab=None):
    """
    train_embeddings: read the tsv of training data and calculate embeddings of training tweet
    
    Returns:
        embedding: trained embedding
    """
    train = pd.read_csv(DATA_DIR + 'olid-training-v1.0.tsv', sep='\t')
    
    # convert values into list to pass into training function
    documents = list(train.tweet.values)
    labels = list(train.subtask_a.values)
#    documents = ["hello yucky", "hello ok"]
#    labels = ["off", "not"]
    return np.array([tweet_to_embedding(embedding, tweet, vocab) for tweet in documents]), labels


def find_nearest(array, value):
    idx = (np.abs(array - value)).sum(axis=1).argmin()
    return idx


def evaluate(embedding, training_embeddings, labels, vocab=None):
    """
    evaluate: evaluate all of the test documents from the csv's and print the results
    
    Inputs:
        embedding: embedding to test
    """
    test_labels_df = pd.read_csv(DATA_DIR + 'labels-levela.csv', sep='\t', header=None, names=["id_label"])
    test_labels = [val.split(",")[1] for val in test_labels_df.id_label.values]
    
    test_docs_df = pd.read_csv(DATA_DIR + 'testset-levela.tsv', sep='\t')
    test_docs = list(test_docs_df.tweet.values)
    
    preds = []
    for d in test_docs:
        preds.append(labels[find_nearest(training_embeddings, tweet_to_embedding(embedding, d, vocab))])
        
    return calc_results(test_labels, preds)
    

def eval_sgns_embedding(embedding_path):
    print(embedding_path[-12:])
    embedding = load_sgns_embedding(embedding_path)
    training_embeddings, labels = train_embeddings(embedding)
    evaluate(embedding, training_embeddings, labels)


def eval_glove_embeddings(embedding_path):
    print(embedding_path[-12:])
    embedding, vocab = load_glove_embedding(embedding_path)
    vocab = vocab.token2index.keys()
    training_embeddings, labels = train_embeddings(embedding, vocab)
    return evaluate(embedding, training_embeddings, labels, vocab)


def cosine_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))


def msw(a, n, m, vocab):
    top_words = set(np.argsort([cosine_sim(m[token], m[a]) for token in vocab])[-n:])

    return top_words


def j_at_n(A, models, n, vocab):
    jaccard_sums = 0
    for a in tqdm(A):
        intersection = set(range(len(vocab))).intersection(*[msw(a, n, m, vocab) for m in models])
        union        = set().union(       *[msw(a, n, m, vocab) for m in models])
        jaccard_sums += len(intersection) / len(union)
    return jaccard_sums / len(A)


def get_top_words():
    vocab = load_glove_embedding("/csse/users/grh102/Documents/cosc442/unsupervised-systems/nlp_glove_implementation/glove/data/vectors1.txt")[1]
    top_words_indices = np.argsort(vocab.token_counts)[-1000:]
    top_words = [vocab.index2token[index] for index in top_words_indices]
    return top_words, vocab


def intrinsic_eval_sgns_embedding():
    sgns_base_dir = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/word2vec-master/vectors"
    top_words, vocab = get_top_words()

    models = []
    for i in range(1, 6):
        model, vocab = load_sgns_embedding(sgns_base_dir + str(i) + ".bin")
        models.append(model)

    j = j_at_n(top_words, models, 10, vocab)
    print("SGNS", j)
    

def intrinsic_eval_glove_embeddings():
    glove_base_dir = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/nlp_glove_implementation/glove/data/vectors"
    top_words, vocab = get_top_words()

    models = []
    for i in range(1, 6):
        models.append(load_glove_embedding(glove_base_dir + str(i) + ".txt")[0])

    j = j_at_n(top_words, models, 10, vocab.token2index.keys())
    print("GloVe:", j)



if __name__ == "__main__":
    intrinsic_eval_sgns_embedding()
    intrinsic_eval_glove_embeddings()

    sgns_base_dir = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/word2vec-master/vectors"
    sgns_results = []
    for i in range(1, 6):
        sgns_results.append(eval_sgns_embedding(sgns_base_dir + str(i) + ".bin"))

    glove_base_dir = "/csse/users/grh102/Documents/cosc442/unsupervised-systems/nlp_glove_implementation/glove/data/vectors"
    glove_results = []
    for i in range(1, 6):
        glove_results.append(eval_glove_embeddings(glove_base_dir + str(i) + ".txt"))

    print("SGNS: ")
    sgns_f1  = [result[0] for result in sgns_results]
    print(sgns_f1)
    print("Mean:", mean(sgns_f1))
    print("Variance:", variance(sgns_f1))

    print("GloVe: ")
    glove_f1 = [result[0] for result in glove_results]
    print(glove_f1)
    print("Mean:", mean(glove_f1))
    print("Variance:", variance(glove_f1))
