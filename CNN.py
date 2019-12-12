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


    #The two inputs:

    #img_input is 12 images representing
    #cloud cover over a single hour
    #at five minute intervals
    img_input = keras.models.Sequential()

    #solar_input is a single value from PySolar
    #that is the expected solar irradiance 
    #under clear sky conditions
    solar_input = keras.models.Sequential()

    #First define the shape of each of the inputs
    img_input.add(Input(shape=(12,100,150,3)))
    solar_input.add(Input(shape=(1)))

    #Each of these are time distributed convolutional layers
    #this means that it will take the 12 images and keep them in order
    #and basically assign feature maps to each of the 12 images
    img_input.add(TimeDistributed(Conv2D(32, kernel_size=(7,7),
                                        strides=(2,2), padding='valid',
                                        activation='relu')))
    img_input.add(TimeDistributed(Conv2D(64, kernel_size=(3,3),
                                        strides=(2,2), padding='valid', 
                                        activation='relu')))
    img_input.add(TimeDistributed(Conv2D(128, kernel_size=(3,3), 
                                        strides=(2,2), padding='valid', 
                                        activation='relu')))


    #Flatten the feature maps to pass through the bidirectional layer
    img_input.add(TimeDistributed(Flatten()))
    img_input.add(Bidirectional(LSTM(128, activation='relu')))

    #Pass all of the feature maps through a single dense layer with
    # a lower number of nodes so as not to drown out the 
    #input from PySolar
    img_input.add(Dense(16,activation='relu'))


    #concatenate the output above with the solar_input
    #and pass this through 2 dense layers
    concat = keras.layers.concatenate([img_input.output,solar_input.output])
    concat = Dense(16,activation='relu')(concat)
    concat = Dense(1)(concat)

    model = keras.models.Model(inputs=[img_input.input,solar_input.input],outputs=concat)

    model.summary()

    model.compile(loss='mse',optimizer='adam')
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


