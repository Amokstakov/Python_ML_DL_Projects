# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:27:07 2020

@author: Amoks
"""

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K


import cv2
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter

df = pd.read_csv(r'C:/Users/Amoks/Desktop/Personal/MachineLearning/DataSets/predicting-a-pulsar-star/pulsar_stars.csv')

X = df.drop('target_class',axis=1)
y = df['target_class']

X = preprocessing.scale(X)

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=50)

x_train = np.array(x_train) 
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)   

#Create Simple NN
def tune_model(hp):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(hp.Int('dense_units',
                                             min_value=32,
                                             max_value=256,
                                             step=32),
                                             activation='relu'))
    
        
    for i in range(hp.Int('n_layers', 1,4)):
        model.add(tf.keras.layers.Dense(hp.Int(f'dense_{i}_units',
                                             min_value=32,
                                             max_value=256,
                                             step=32),
                                             activation='relu'))
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    tune_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2)

tuner.search(
    x=x_train,
    y=y_train,
    epochs=3,
    batch_size=32,
    validation_data=(x_test,y_test)
    )
    
print(tuner.get_best_hyperparameters()[0].values)
print(tuner.get_best_models()[0].summary)

def create_model():
    model = keras.models.Sequential([
            keras.layers.Dense(160, activation='relu'),
            keras.layers.Dense(224, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(96, activation='relu'),
            keras.layers.Dense(1,activation='sigmoid')
            ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = create_model()

history = model.fit(x_train,y_train,batch_size=32,epochs=30,verbose=True)
scores = model.evaluate(x_test,y_test,verbose=1)

print(scores[1])

# 0.9793

#Use Grid Search for each

#Random Forerst

from sklearn.ensemble import RandomForestClassifier #import the model library
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0) # sitting model parameters
print("test accuracy: {} ".format(rf.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(rf.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set


#Logistic Regression

from sklearn import linear_model #import the model library
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 500,solver='lbfgs') # sitting model parameters
print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set


#Decision Tree Classifier

from sklearn import tree #import the model library
dt = tree.DecisionTreeClassifier() # sitting model
print("test accuracy: {} ".format(dt.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(dt.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set


#KNN

from sklearn.neighbors import KNeighborsClassifier #import the model library
neigh = KNeighborsClassifier(n_neighbors=3) # sitting model parameters
print("test accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set


#SVM

from sklearn import svm #import the model library
svm = svm.SVC(gamma='scale') # sitting model parameters
print("test accuracy: {} ".format(svm.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(svm.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set

#XGBoost