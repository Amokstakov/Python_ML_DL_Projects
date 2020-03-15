# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:15:59 2020

@author: Sasha
"""

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K


#import tensorflow as tf


K

K.backend()

import os 
import time
import pickle
import random
import numpy as np
import pandas as pd
#import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard


seq_len = 60 #how long of a preceeding sequence to collect for RNN
future_to_predict = 3 #how far into the future are we trying to predict
ratio_to_predict = "LTC-USD" #what we are predicting 

data_dir = 'C:/Users/Amoks/Desktop/Personal/MachineLearning/DataSets/crypto_data/'


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


#Get Close and volume for each separate category, and then we join them all because of their shared axis of time
def create_df():
    main_df = pd.DataFrame()
    coins = ['BTC-USD','LTC-USD','BCH-USD','ETH-USD']
    for coin in coins:
        print(coin)
        path = os.path.join(data_dir, (coin+'.csv'))
        df = pd.read_csv(path, names=['time','low','high','open','close','volume'])
        df.rename(columns={'close': f"{coin}_close", 'volume':f"{coin}_volume"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{coin}_close",f"{coin}_volume"]] 
        if len(main_df) == 0:
            main_df = df
        else: 
            main_df = main_df.join(df)
    
    main_df.fillna(method='ffill', inplace=True)
    main_df.dropna(inplace=True)
    return main_df

main_df = create_df()

##Very important function that requires some explaination. WE are just shifting the future price from 3 units down into a new column
#LTC_current Pice LTC_future_price
##LTC_future_price is just the pric of LTC current from 3 minutes or 3 future_to_predict units down
main_df['future'] = main_df[f"{ratio_to_predict}_close"].shift(-future_to_predict)

#Maps the function (calls the function) into a list
main_df['target'] = list(map(classify, main_df[f"{ratio_to_predict}_close"], main_df['future']))


#super important that time is sorted
times = sorted(main_df.index.values)
last_5_pct = times[-int(0.05*len(main_df))]

validation_main_df = main_df[(main_df.index >= last_5_pct)]
validation_main_df_2 = main_df[main_df.index.get_loc(last_5_pct):]

main_df = main_df[:main_df.index.get_loc(last_5_pct)]

def preprocess_df(df):
    df = df.drop('future', 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=seq_len)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == seq_len:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), np.array(y)  # return X and y...and make X a numpy array!
            


x_train, y_train = preprocess_df(main_df)
x_test, y_test = preprocess_df(validation_main_df_2)


##Build actual model
epochs = 10
batch_size = 64
name = f"{seq_len}-SEQ-{future_to_predict}-PRED-{int(time.time())}"

def build_model():
    model = keras.models.Sequential([
            keras.layers.LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True, activation='tanh'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            
            keras.layers.LSTM(128, return_sequences=True, activation='tanh'),
            keras.layers.Dropout(0.1),
            keras.layers.BatchNormalization(),
            
            keras.layers.LSTM(128, activation='tanh'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
        
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(2, activation='softmax')
            
            ])
    
    opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model


model = build_model()
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))








