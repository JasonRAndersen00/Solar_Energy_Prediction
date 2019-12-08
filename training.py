###########################################################
# training.py
# Author: Jason Andersen
#
# trainign.py is used for the implementation and training of
# the CNN

# To function the following dependencies need to be installed:
#       PySolar
#       tensorflow and Keras
#       numpy
#       matplotlib
#       pytz
#       netCDF4 :  conda install -c conda-forge netcdf4
############################################################



import numpy as np
import ClearSkyInput
import helperFunctions
import CNN
import pickle
import tensorflow.keras as keras
import os


year = 2018
days = list()
x1 = list()


#read in full days from the specified directory and extract correct hours
fullDaysDirectory = './data/cloud/2018/FullDays/'
for filename in os.listdir(fullDaysDirectory):
    if filename.endswith(".pickle"): 
        days.append(int(os.path.splitext(filename)[0]))
        file = open(fullDaysDirectory + filename,'rb')
        x1 += pickle.load(file, encoding='latin1')[7:18]
        file.close()

#read in partial days and add to list
partialDaysDirectory = './data/cloud/2018/PartialDays/'
for filename in os.listdir(partialDaysDirectory):
    if filename.endswith(".pickle"): 
        days.append(int(os.path.splitext(filename)[0]))
        file = open(partialDaysDirectory + filename,'rb')
        x1 += pickle.load(file, encoding='latin1')
        file.close()

print(days)

# put into numpy array
x1 = np.asarray(x1)
print('x1 shape: ', x1.shape)

#get clear sky input from PySolar for the specified days and year, then reshape
x2 = ClearSkyInput.getClearSkyIrradianceVariableDays(41.752032, -111.793835,year,days)
x2 = np.asarray(x2).reshape(-1,1)

#get the correct ground truth for the specified days .
y = helperFunctions.getSolarGroundTruthVariableDays(days,year-2000,'data/solarEnergy.csv')


print('x2 shape: ', x2.shape)
print('y shape: ', y.shape)


#Train the network
epochs = 50
batch_size = 32
#number_dense_nodes is a variable amount of dense nodes that the CNN will be passed through
#before concatenating with the Clear Sky Input. Too high drowns out the Clear sky input
# Too low causes the CNN to not be used at all. 
number_dense_nodes = 16
directoryToLoadNetwork = './model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes)
directoryToSaveNetwork = './model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes)
directoryToSaveLogs = 'logs/{0}'.format(number_dense_nodes)

# only one of the following 2 lines needs to be uncommented. The first line
#will generate a new CNN the second will load an existing CNN
model = CNN.define_model(number_dense_nodes)
# model = keras.models.load_model(directoryToLoadNetwork)
CNN.train_model(model,x1,x2,y,epochs, batch_size,directoryToSaveLogs)
model.save(directoryToSaveNetwork)
keras.backend.clear_session()
