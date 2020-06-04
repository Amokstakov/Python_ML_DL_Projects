import numpy as numpy
import pickle
import tensorflow as tf
from tf.keras.callbacks import ModelCheckpoint

##layers
from tf.keras.models import Model
from tf.keras.layers import Input
from tf.keras.layers import Dense
from tf.keras.layers import LSTM
from tf.keras.layers import Embedding
from tf.keras.layers import Dropout
from tf.keras.layers.merge import add

#load data
def load_data(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#load predefine list of photo identifies => returns a list of just image titles
def load_set(filename):
    doc = load_data(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 0:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return dataset

#load clean descriptions
def load_clean_descriptions(filename, dataset):
    doc = load_data(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'start' + ' '.join(image_desc) + 'end'
            descriptions[image_id].append(desc)
    return descriptions

#load photo features
def load_photo_features(filename, dataset):
    print("um wtf")
    all_features = pickle.load(open(filename, 'rb'))
    #filter features
    features = {k: all_features[k] for k in dataset} 
    return features

#conver dict of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    all_desc = to_lines(descriptions)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer

#calculate max len of vocab for embedding layer
def max_len(descriptions):
    all_desc = to_lines(descriptions):
    return max(len(d.split()) for d in all_desc)

#create seqeucnes of images, input seqeucnes and output words for an image
def create_sequences(tokenizer, max_len, descriptions, photos, vocab_size):
    x1, x2, y = list(), list(), list()
    #walk through image identifier
    for key, desc_list in descriptions.items():
        #walk through each description of the image
        for desc in desc_list:
            #encode the sequence
            seq = tokenizer.texts_to_sequence([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_len)[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                #store
                x1.append(photos[key][0])
                x2.append(in_seq)
                y.append(out_seq)
    return array(x1), array(x2), array(y)

def define_model(vocab_size, max_len):
    #feature Extraction Model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)
    #sequence model
    inputs2 = Input(shape=(max_len))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #decoder model
    de1 = add([fe1, se1])
    de2 = Dense(256, activation="relu")(de1)
    outputs = Dense(vocab_size, activation="softmax")(de2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy",optimizer="adam")
    return model

#execution

##training set
filename = "/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/photo_captioning/FlickerText/Flickr_8k.trainImages.txt"
train = load_set(filename)

train_descriptions = load_clean_descriptions('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/WordEmbeddings/mapping.txt',train)

train_features = load_photo_features('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/WordEmbeddings/features.pkl',train)

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_len(train_descriptions)

x1,x2,y = create_sequences(tokenizer, max_len, train_descriptions, train_features, vocab_size)


##testing set
filename = "/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/photo_captioning/FlickerText/Flickr_8k.devImages.txt"
test = load_set(filename)

test_descriptions = load_clean_descriptions('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/WordEmbeddings/mapping.txt',test)

test_features = load_photo_features('/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/WordEmbeddings/features.pkl', test)

x1Test,x2Test,yTest = create_sequences(tokenizer, max_len, test_descriptions, test_features, vocab_size)

model = define_model(vocab_size, max_len)

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit([x1, x2], y, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([x1test, x2test], ytest))





