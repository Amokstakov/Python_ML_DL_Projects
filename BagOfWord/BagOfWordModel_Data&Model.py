import string
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load tokens
def load_tokens(vocabpath):
    file = open(vocabpath, 'r')
    tokens = file.read()
    file.close()
    return tokens

def clean_doc_with_tokens(doc, vocab):
    tokens = doc.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word in vocab]
    tokens = ' '.join(tokens)
    return tokens


def process_docs(path, vocab, train):
    documents = list()
    for file in os.listdir(path):
        if train and file.startswith('cv9'):
            continue
        if not train and not file.startswith('cv9'):
            continue
        path_ = path + '/' + file
        tokens = load_tokens(path_)
        tokens = clean_doc_with_tokens(tokens, vocab)
        documents.append(tokens)
    return documents


#load tokens
vocabpath = '/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/BagOfWord/token_list.txt'
vocab = load_tokens(vocabpath)
vocab = vocab.split()
vocab = set(vocab)


positive_ = process_docs('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/txt_sentoken/pos', vocab, True)
negative_ = process_docs('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/txt_sentoken/neg', vocab, True)
training_set = positive_ + negative_

tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_set)

#sequence encode
encoded = tokenizer.texts_to_sequences(training_set)
#pad len
max_len = max([len(s.split()) for s in training_set])
x_train = pad_sequences(encoded, maxlen=max_len, padding="post")
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

#model definition
#Embedding requires 3 params, vocab set len, vector space dimension, max document len
vocab_size = len(tokenizer.word_index) + 1

def create_train_model(x,y,vocab_size, max_doc_len):

