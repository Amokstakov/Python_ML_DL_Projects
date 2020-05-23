#Data Preparation
##Headings, English spelling, punctuation, names, long monologues, quoted dialogs

#Model Sequence
##Input sequence will be 50 words, output will be one word

###############
import string
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np

#load the file
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#clean doc for tokens
def clean_doc(text):
    text = text.replace('--',' ')
    tokens = text.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

#save our cleaned tokens in a separate file
def save_sequences(sequences, filename):
    data = '\n'.join(sequences)
    file = open(filename, 'w')
    file.write(data)
    file.close()

#Defie sequence inputs and outputs
def sequences(inputLen, token):
    sequences = list()
    for i in range(inputLen, len(token)):
        seq = tokens[i-inputLen:i]
        seq = ' '.join(seq)
        sequences.append(seq)
    return sequences

input_path = 'data/Plato.txt'
#load the data
text = load_doc(input_path)
tokens = clean_doc(text)
sequences = sequences(51, tokens)

output_name = "Plato_sequences.txt"
save_sequences(sequences, output_name)

#load sequences after saving

doc = load_doc(output_name)
lines = doc.split('\n')


#Integer encode sequence of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1

#separate into input and output
sequences = np.array(sequences)
X,y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


#Create the model

def build_fit_model(X,y,vocab_size, seq_length):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 50, input_length=seq_length),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(vocab_size, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])
    model.fit(X,y,batch_size=160,epochs=20)

    return model


model = build_fit_model(X,y,vocab_size,seq_length)
model.save('model.h5')
dump(tokenizer, open('tokenizer.pkl','wb'))