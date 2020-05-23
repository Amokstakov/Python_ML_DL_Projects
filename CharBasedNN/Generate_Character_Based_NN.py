from pickle import load
import tensorflow as tf
import sys

def generate_seq(model,mapping, seq_len, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_len, truncating="pre")
        encoded = tf.keras.utils.to_categorical(encoded,num_classes=len(mapping))
        #encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        ypred = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == ypred:
                out_char = char
                break
        in_text += char
    return in_text

model = tf.keras.models.load_model('model.h5')
mapping = load(open('mapping.pkl','rb'))


print(generate_seq(model, mapping, 10, 'sing a son',20))
print(generate_seq(model, mapping, 10, 'king was i',20))
print(generate_seq(model, mapping, 10, 'hello worl',20))