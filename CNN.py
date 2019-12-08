###########################################################
# CNN.py
# Author: Jason Andersen
#
# CNN.py is used for the implementation the CNN
############################################################
import tensorflow as tf
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns 

import datetime
from pytz import timezone

from tensorflow.keras.callbacks import TensorBoard
import numpy as np  
# from skimage import transform 

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, TimeDistributed, Flatten, Bidirectional, LSTM

import helperFunctions
import ClearSkyInput 

sns.set()
sns.set_palette("colorblind")
def define_model(dense_nodes):

    img_input = keras.models.Sequential()
    solar_input = keras.models.Sequential()

    #shape=(numImages, width,height,channesl)
    img_input.add(Input(shape=(12,100,150,3)))
    solar_input.add(Input(shape=(1)))

    #dialation parameter, dialation=2 or 4
    img_input.add(TimeDistributed(Conv2D(32, kernel_size=(7,7), strides=(2,2), padding='valid', activation='relu')))
    img_input.add(TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu')))
    img_input.add(TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu')))
    ##add more as needed



    img_input.add(TimeDistributed(Flatten()))
    img_input.add(Bidirectional(LSTM(128, activation='relu')))#train faster run slower
    img_input.add(Dense(dense_nodes,activation='relu'))
    # img_input.add(Bidirectional(LSTM(128, activation='relu', dropout=0.5)))#train faster run slower
    # img_input.add(Bidirectional(LSTM(128, activation='relu', dropout=0.5)))#train faster run slower



    # solar_input = Dense(16,activation='relu')(solar_input) #If uncomment this, comment out the next dense layer
    concat = keras.layers.concatenate([img_input.output,solar_input.output])
    concat = Dense(16,activation='relu')(concat)
    concat = Dense(1)(concat)

    model = keras.models.Model(inputs=[img_input.input,solar_input.input],outputs=concat)

    model.summary()

    model.compile(loss='mse',optimizer='adam')#,learning_rate=0.01)
    return model




def train_model(model, x_image,x_solar,y,epochs,batch_size,logDir):
    
    model.fit(
        x = [x_image,x_solar],
        y = y,
        batch_size = batch_size,
        epochs =epochs,
        verbose=1,
        callbacks=[TensorBoard(log_dir=logDir)],
        validation_split=0.2,
        shuffle=True)


