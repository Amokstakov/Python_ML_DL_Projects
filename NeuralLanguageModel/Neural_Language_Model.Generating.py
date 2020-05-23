from pickle import load
import tensorflow as tf
import random
from keras.preprocessing.sequence import pad_sequences

#load doc into memory
def load_doc(path):
    file = open(path, 'r')
    text = file.read()
    file.close()
    return text

def generate_sequence(model, tokenizer, seq_len, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_len, truncating='pre')
        ypred = model.predict_classes(encoded,verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == ypred:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

#load cleaned text sequences
filename = 'Plato_sequences.txt'
text = load_doc(filename)
lines = text.split('\n')
seq_len = len(lines[0].split()) -1 

model = tf.keras.models.load_model('model.h5')

tokenizer = load(open('tokenizer.pkl', 'rb'))

seed_text = lines[random.randint(0,len(lines))]
print(seed_text)

generated = generate_sequence(model, tokenizer, seq_len, seed_text, 50)
print(generated)