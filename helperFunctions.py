###########################################################
# helperFunctions.py
# Author: Jason Andersen
#
# helperFunctions.py is used to get the Solar Ground truth from
# the specified CSV file
############################################################
import netCDF4
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pickle

def getDaySolarGroundTruth(filename,year,day):
    solarEnergy = list()
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            # print(row)
            # print(row[2][1:3])
            thisYear = int(row[0])
            # print('thisYear: ', thisYear)
            if thisYear == year:

                thisday = int(row[1])
                if thisday == day:
                    solarEnergy.append(float(row[3]))
    return solarEnergy[7:18]

def getSolarGroundTruthVariableDays(days,year,filename):
    data = list()
    for day in days:
        data = data + (getDaySolarGroundTruth(filename,year,day))

    return np.asarray(data).reshape(-1,1)#.astype(np.float16)

if __name__ == "__main__":
    days = (2,4)
    year = 18
    data = getSolarGroundTruthVariableDays(days,year,'data/solarEnergy.csv')
    print(data.shape)




