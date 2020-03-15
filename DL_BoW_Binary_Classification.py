# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:14:47 2020

@author: Amoks
"""

from string import punctuation
import numpy as np
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import tensorflow as tf


'''
The process is to take a bunch of documents in each respective folder and take
the amount of unique vocabulary in each one and store it:
    steps:
        walk through all the documents in each path: Done
        load the document
        clean to tokenize the document
        update the vocab
'''
#function to create and process through the text and extract tokens
def clean_docs(doc):
    tokens = doc.split()
    #remove punctuation
    table = str.maketrans('','',punctuation)
    tokens = [w.translate(table) for w in tokens]
    #remove non alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    #filter out all the stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    #filter out all the short tokens
    tokens = [w for w in tokens if len(w) > 1]
    return tokens

#Loads each document, reads it and takes the text from it 
def load_docs(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

#combine loading and cleaning functions into bride function to update vocab
def load_clean(path,vocab):
    doc = load_docs(path)
    tokens = clean_docs(doc)
    vocab.update(tokens)
    
def cycle_docs(docs, vocab):
    for file in os.listdir(docs):
        if file.startswith('cv9'):
            continue
        path = docs + '/' + file
        load_clean(path,vocab)
        
     

vocab = Counter()

positive_path = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\txt_sentoken\pos'
negitive_path = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\txt_sentoken\neg'

cycle_docs(positive_path, vocab)
cycle_docs(negitive_path, vocab)

#total amount of words in vocab
print(len(vocab))

#top 50
print(vocab.most_common(50))

#We can remove all the words that do not appear frequently 
min_ = 2
vocab = [k for k,c in vocab.items() if c >= min_]

def save_vocab(vocab, filename):
    data = '\n'.join(vocab)
    filename = open(filename, 'w')
    filename.write(data)
    filename.close()
    
save_vocab(vocab, 'vocab.txt')


####Now that we have developped our vocab, we need to go through each document in the folders
# and convert them, and clean from if the words are not in vocab.

#this function cleans the reviews for words not in vocab
def doc_to_vector(path,vocab):
    doc = load_docs(path)
    tokens = clean_docs(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

#this function goes through each file and performs doc_to_vector
def cycle_docs_(docs, vocab, flag):
    lines = list()
    for file in os.listdir(docs):
        if flag and file.startswith('cv9'):
            continue
        if not flag and not file.startswith('cv9'):
            continue
        path = docs + '/' + file
        line = doc_to_vector(path,vocab)
        lines.append(line)
    return lines
    
#load the vocab
vocab_filename = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\vocab.txt'
vocab = load_docs(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
print(len(vocab))

positive_path = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\txt_sentoken\pos'
negitive_path = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\txt_sentoken\neg'

positive_lines = cycle_docs_(positive_path, vocab, True)
negative_lines = cycle_docs_(negitive_path, vocab, True)



###Now we need to create the movie rewviews to BoW vectors
tokenizer = Tokenizer()
docs = positive_lines + negative_lines
tokenizer.fit_on_texts(docs)

#encode training data
x_train = tokenizer.texts_to_matrix(docs, mode='freq')
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
print(x_train.shape)

#load testing data set
positive_testing_lines = cycle_docs_(positive_path, vocab, False)
negative_testing_lines = cycle_docs_(negitive_path, vocab, False)

docs_test = positive_testing_lines + negative_testing_lines
x_test = tokenizer.texts_to_matrix(docs_test, mode='freq')
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])
print(x_test.shape)


###Developping the NN model
#We will create our sentiment analysis for classification on this problem

n_words = x_test.shape[1]

def build_fit_model(x,y):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(50, input_shape=(n_words,),activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x,y,epochs=50,verbose=2)
    
    return model

model = build_fit_model(x_train,y_train)   

loss, accuracy = model.evaluate(x_test,y_test)



###We will not create a function that is able to predict on new reviewsd.
#to do this, we need to clean the incoming new reviews just like we did
#for thr training and testing data

def clean_new_prediction(text, vocab, tokenizer, model):
    tokens = clean_docs(text)
    tokens = [w for w in tokens if w in vocab]
    line =  ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix(line,mode='freq')
    prediction = model.predict(encoded, verbose=1)
    return round(prediction[0,0])


sample_text = 'I really dont like anything and I feel like this is shitty'

prediction = clean_new_prediction(sample_text, vocab, tokenizer, model)   
print(prediction)


