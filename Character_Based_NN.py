# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:57:35 2020

@author: Amoks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

'''
The point of this script is to create a model that can predict and generate
the next character in a word based on the sequences of characters/words
before it 
Steps to completing this are:
    prepare the text data
        Our data has to be prepared with sequences in mind. We need to pas
        a certain amount of sequences as our input to our model so that
        the model can give a valid output, and once it does we can apppend
        that output into the next input sequence
    traing and build the model
    generate the text
'''

##Data Prep
#Load the data

def load_data(data):
    file = open(data,'r')
    text = file.read()
    file.close()
    return text

path = r'C:\Users\Amoks\Desktop\Personal\MachineLearning\DataSets\rhyme.txt'

raw_text = load_data(path)


#clean the data
def clean_data(raw_text):
    tokens = raw_text.split()
    raw_text = ' '.join(tokens)
    return raw_text

raw_text = clean_data(raw_text)

#Now we need to create our training data of all of our sequennces
#the example I am following is using sequence length of 10 for inputs, and
#outputting one value for the label of those 10 character sequences


#this function breaks up our raw text into multiple sequences starting at 0
#and moving forward until the entire len of the raw text is done
#this works by using our n_sequence, and making sure we are indexing our 
#raw_text accurately in the data to make the proper sequences needed
def sequence_data(data):
    sequences = list()
    n_sequence = 10
    for i in range(10, len(data)):
        sequence = data[i-n_sequence:i+1]
        sequences.append(sequence)
    return sequences

sequences = sequence_data(raw_text)

###Training the model
#We first need to go back to the orignal text and assign every unique chracter an interger value
chars = sorted(list(set(raw_text)))
mapping = dict((c,i) for i,c in enumerate(chars))

sequence_list = list()
for line in sequences:
    encoded_seq = [mapping[char] for char in line]
    sequence_list.append(encoded_seq)
        
#Our sequence_list now contains all of our sequences in the form of intergers
vocab_size = len(mapping)


###We now can split our sequences into input and output style of data
sequence_list = np.array(sequence_list)        

x_train,y_train = sequence_list[:,:-1], sequence_list[:,-1]

#Now we can transform our data with OHE. 
sequence_list = [tf.keras.utils.to_categorical(x, num_classes=48) for x in x_train]        

x_train = np.array(sequence_list)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=38)

##Define and fit our model
def build_fit_model(x,y):
    model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(75, input_shape=(x_train.shape[1],x_train.shape[2])),
            tf.keras.layers.Dense(38, activation='softmax')
            ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x,y,epochs=100,verbose=2)
    
    return model

model = build_fit_model(x_train,y_train)


###Generate Characters - DOES NOT WORK
def generate_seq(model, mapping, seed, text, chars):
    in_text = text
    for _ in range(chars):
        encoded = [mapping[char] for char in in_text]
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seed, truncating='pre')
        encoded = tf.keras.utils.to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        prediction = model.predict_classes(encoded, verbose=0)
        out_char = ""
        for char, index in mapping.items():
            if index == prediction:
                out_char = char 
                break
        in_text += char
    return in_text

print(generate_seq(model, mapping, 10, 'Sing a son', 20))






        
        
        
