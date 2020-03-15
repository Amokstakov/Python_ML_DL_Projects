# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:20:38 2020

@author: Amoks
"""

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K

K
K.backend()

import os
import cv2
import time
import keras
import pickle
import random
import numpy as np
import pandas as pd
import plaidml.keras
import keras.backend as K
plaidml.keras.install_backend()
import matplotlib.pyplot as plt
from sklearn import preprocessing
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"



class_names = ['Banana','Cherry', 'Tomato', 'Onion']
data_dir = r'C:/Users/Amoks/Desktop/Personal/MachineLearning/DataSets/fruits/fruits-360_dataset/fruits-360/Training/Sub_set/'

img_size= 75

def create_training_data():
    img_size= 75
    training_data = []
    for category in class_names:

        path = os.path.join(data_dir,category)
        class_num = class_names.index(category)  

        for img in os.listdir(path): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append([new_array, class_num]) 
            except Exception as e:
                print(e)
    return training_data


training_data = create_training_data()

print(len(training_data[0]))

random.shuffle(training_data)

X = []
y = []

for img, label in training_data:
    X.append(img)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size,img_size, 1)

X = X / 255.0

def create_model():
    model = keras.models.Sequential([
            keras.layers.Conv2D(256, (3,3), input_shape=X.shape[1:], activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            
            keras.layers.Conv2D(256, (3,3),activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            
            keras.layers.Flatten(),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(4, activation='softmax')
            ])
    
    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    
    return model

model = create_model()

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)


test_set = []
test_data_dir_1 = 'C:/Users/Amoks/Desktop/Personal/MachineLearning/DataSets/fruits/fruits-360_dataset/fruits-360/test-multiple_fruits/'

for img in os.listdir(test_data_dir_1):
    if 'Bananas' in img:
        img_array = cv2.imread(os.path.join(test_data_dir_1,img) ,cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size,img_size))
        


test_set = np.array(test_set).reshape(-1, img_size,img_size, 1)

prediction = model.predict([test_set])



testbanana = cv2.imread('C:/Users/Amoks/Desktop/banana.jpg', cv2.IMREAD_GRAYSCALE)
testbanana = cv2.resize(testbanana, (img_size,img_size))
testbanana = testbanana / 255.0
testbanana = np.array(testbanana).reshape(-1, img_size, img_size, 1)

prediction = model.predict(testbanana)


