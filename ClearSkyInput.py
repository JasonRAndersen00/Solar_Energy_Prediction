
###########################################################
# ClearSkyInput.py
# Author: Jason Andersen
#
# ClearSkyInput.py contains wrapper functions for getting the
# correct days from PySolar for US Mountain Time
############################################################
# #This code was based on the following url: 
# https://earthscience.stackexchange.com/questions/14491/how-to-calculate-the-solar-radiation-at-any-place-any-time

import datetime
from pytz import timezone
import matplotlib.pyplot as plt
import pysolar
import numpy as np

def getClearSkyIrradiance(lat,lon,start,numhours):
    # start = datetime.datetime(year,month,day,hour,tzinfo=datetime.timezone.utc)

    # Calculate radiation every hour for 90 days

    dates, altitudes_deg, radiations = list(), list(), list()
    for ihr in range(numhours):
        date = start + datetime.timedelta(hours=ihr)
        altitude_deg = pysolar.solar.get_altitude(lat,lon,date)
        if altitude_deg <= 0:
            radiation = 0.
        else:
            radiation = pysolar.radiation.get_radiation_direct(date,altitude_deg)
        dates.append(date)
        altitudes_deg.append(altitude_deg)
        radiations.append(radiation)
    return np.asarray(radiations).astype(np.float16).reshape(-1,1)

def getClearSkyIrradianceSingleDay(lat,lon,start):
    radiations = list()
    for ihr in range(24):
        date = start + datetime.timedelta(hours=ihr)
        altitude_deg = pysolar.solar.get_altitude(lat,lon,date)
        if altitude_deg <= 0:
            radiation = 0.
        else:
            radiation = pysolar.radiation.get_radiation_direct(date,altitude_deg)

        radiations.append(radiation)
    return radiations[7:18]

def getClearSkyIrradianceVariableDays(lat,lon,year, days):
    
    data = list()
    for day in days:
        d =  datetime.datetime.strptime('{} {}'.format(day, year),'%j %Y')
        startMnt = datetime.datetime(year,d.month,d.day,00,tzinfo=timezone('US/Mountain'))
        data = data + (getClearSkyIrradianceSingleDay(lat,lon,startMnt))
        pass
    return data



