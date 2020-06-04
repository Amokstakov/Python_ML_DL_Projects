import os 
import numpy as np
import pickle 
import tensorflow as tf 
import matplotlib.pyplot as plt

#extract features from each photo in a directory
def extract_features(directory):
    #load the VGG16 Model
    model = tf.keras.applications.vgg16.VGG16()
    #re-structure the model
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    #summarize
    #print(model.summary())
    #extract features from the directory
    features = dict()
    for name in os.listdir(directory):
        #load an image from file
        filename = directory + "/" + name
        image_ = tf.keras.preprocessing.image.load_img(filename, target_size=(224,224))

        #conver image pixels to array
        image_ = tf.keras.preprocessing.image.img_to_array(image_)

        #reshape image
        image_ = image_.reshape((1, image_.shape[0], image_.shape[1], image_.shape[2]))

        #need to prepare the image for VGG
        image_ = tf.keras.applications.vgg16.preprocess_input(image_)
        #since we removed the last classification layer, at this point we should get our features returned
        feature = model.predict(image_, verbose=1)
        image_id = name.split('.')
        key = image_id[0]
        #image_id = image_id[0]
        features[key] = feature
    return features

directory = '/home/sasha/Documents/Personal/Python/Python_ML_DL_Projects/data/photo_captioning/Flicker8k_Dataset'
features = extract_features(directory)
print(len(features))
pickle.dump(features, open('features.pkl','wb'))


