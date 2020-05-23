####Character based neural network on a simple nursery
import sys
import numpy as np
from pickle import dump
import tensorflow as tf


#Load function
def load_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#Clean text
def clean_text(text):
    tokens = text.split()
    tokens = ' '.join(tokens)
    return tokens

#Create sequences
def sequences(tokens, length):
    sequences = list()
    for i in range(length, len(tokens)):
        seq = tokens[i-length:i+1]
        sequences.append(seq)
    return sequences

#Save sequences
def save_sequences(sequences, filename):
    data = '\n'.join(sequences)
    file = open(filename, 'w')
    file.write(data)
    file.close()

filename = 'rhyme.txt'

text = load_text(filename)
tokens = clean_text(text)
sequences = sequences(tokens, 10)
output = 'cleaned_rhyme.txt'
save_sequences(sequences, output)

##Train Language Model
filename = 'cleaned_rhyme.txt'
clean_text = load_text(filename)
lines = clean_text.split('\n')

def encode_sequences(clean_text, lines):
    chars = sorted(list(set(clean_text)))
    mapping = dict((c,i) for i,c in enumerate(chars))
    sequences = list()
    for line in lines:
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    vocab_size = len(mapping)
    return sequences, vocab_size, mapping

sequences, vocab_size, mapping = encode_sequences(clean_text, lines)
#vocab size is 38 = 38 unique words in the mapping 

#Split Inputs and Outputs
def split_sequences(sequences, vocab_size):
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    sequences = [tf.keras.utils.to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y =  tf.keras.utils.to_categorical(y, num_classes=vocab_size)
    return X,y 


def buil_and_fit_model(X,y, vocab_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(75, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(vocab_size, activation="softmax")
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X,y, epochs=100, verbose=2)

    return model
X,y = split_sequences(sequences, vocab_size)
model = buil_and_fit_model(X,y,vocab_size)
model.save('model.h5')
dump(mapping, open('mapping.pkl','wb'))




