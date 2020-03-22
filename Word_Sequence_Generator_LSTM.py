# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:40:28 2020

@author: Amoks
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import os 
import string
from keras.preprocessing.text import Tokenizer
import tensorflow as tf


'''
This script will serve as a purpose to create a statistical language model using Deep Learning
1) How to prepare text for developing a word-based model
2) How to design and fit a neural language model
3) how to generate future text with the model assembled and trained

'''

###Data Prep
##We will pick a sequence of 50 characters for our input sequences. This should be long enough for our model to understand the the context, 
##And then provide a generate text that will follow that context

#function to load data from txt file

def load_data(filename):
    file = open(filename, 'r')
    file_contents = file.read()
    file.close()
    return file_contents
    
        
filepath = r"C:\Users\Amoks\Desktop\Personal\DataSets\Plato_The_Republic.txt"
file = load_data(filepath)
#some reason has some strange beginning characters..
file = file[3:]
print(file[:200])


##Clean Text
'''
To clean the text for word-to-word modeling:
    replace - with white spaces
    split words based on the white spaces
    remove all punctuations from words
    remove all words that are not alphabetic
    normalize all words to lower case
'''

def clean_text(text):
    #replace - with spaces
    text = text.replace('--',' ')
    #split words based on white spaces
    tokens = text.split()
    #remove all punctuations from words
    #transitional table
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    #Remove all words that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    #make the words lower case
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = clean_text(file)
        
##Sequence the data
#We will now take the entire lengh of our tokens and sequence it, with 50 input words and 1 output word
length = 51
sequences = list()
for i in range(length,len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)

print(sequences[1])
print(sequences[2])

sequences__ = list()
for i in range(length,len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences__.append(line)



###Training and creating the model
##For our model, our words need to be ints. WE can transform each unique word to correspond to a unique int.
##We can use tokenizer to achieve this

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
#now, every word has a unique corresponding value associated
tokenizer.word_index

#determine the length for our embedding layer
vocab_size = len(tokenizer.word_index) + 1 


###Separate the sequences into inputs and outputs so our model knows what to learn and what to predict
sequences = np.array(sequences)
x,y = sequences[:,:-1], sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
seq_len = x.shape[1]



###Building the model with the parameters defined above
def build_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size,50, input_length=seq_len),
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
            ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

model = build_model()
model.fit(x,y,batch_size=128,epochs=100)

#save the model
model.save('model_LSTM.h5')
model = tf.keras.models.load_model('model_LSTM.h5')


###Generate the text

#Select random seed input from the text
seed_text = sequences__[random.randint(0,len(sequences__))]
print(seed_text)

#encode the seed_text into interger representation
encoded_seed_text = tokenizer.texts_to_sequences([seed_text])[0]
encoded_seed_text.pop()
encoded_seed_text = np.array(encoded_seed_text)
encoded_seed_text = encoded_seed_text.reshape(1,50)

prediction = model.predict_classes(encoded_seed_text)


out_word = ''
for word, index in tokenizer.word_index.items():
	if index == prediction:
		out_word = word
		break
print(out_word)



'''
The function takes in our loaded model, our trained tokenizer, the desired seq_len, our generarted seed text and the amount of words to predict
It first takes the seed text, sets to a variable and encodes to int representation
then it makes sure that the length is 50, if not, it will remove/truncate from the beginning
model makes the word prediction on the inputted 50 words,
we check and transform the in prediction into the word representation
we append to our seed_text variable the new word, and repeat the process. 
'''

##Generate Text function
def generate_text(model, tokenizer,seq_len, seed_text, n_words):
    results = list()
    in_text = seed_text
    for _ in range(n_words):
        #generate a fixed number of words
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        #truncate sequence to a fixed length
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded],maxlen=seq_len, truncating='pre')
        prediction = model.predict_classes(encoded)
        #map predicted word index to word
        out_word=''
        for word,index in tokenizer.word_index.items():
            if index == prediction:
                out_word = word
                break
        #append to input
        in_text += ' ' +out_word
        results.append(out_word)
    return ' '.join(results)

generated = generate_text(model, tokenizer, seq_len, seed_text, 50) 

print(seed_text)
print(generated)





