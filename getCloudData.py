#Based off of code from Brian Blaylock
# URL: https://gist.githubusercontent.com/blaylockbk/d60f4fce15a7f0475f975fc57da9104d/raw/6768f17ce717ae387bc73c8e711e52b6fcef6ac9/download_GOES_AWS.py

import s3fs
import numpy as np
import datetime
from pytz import timezone
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
import pickle


def displayData(filename):
    nc = Dataset(filename, 'r')
    height = 100
    width = 150
    upperLeftRow = 290
    upperLeftCol = 420
    R = nc["CMI_C02"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    # print("R: ", np.min(R), " ", np.max(R))
    G = nc["CMI_C03"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    # print("G: ", np.min(G), " ", np.max(G))
    B = nc ["CMI_C01"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    # print("B: ", np.min(B), " ", np.max(B))


    # Apply range limits for each channel. RGB values must be between 0 and 1
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

     # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)  # apply limits again, just in case.

    fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(1, 4, figsize=(16, 3))

    ax1.imshow(R, cmap='Reds', vmax=1, vmin=0)
    ax1.set_title('Red', fontweight='bold')
    ax1.axis('off')

    ax2.imshow(G, cmap='Greens', vmax=1, vmin=0)
    ax2.set_title('Veggie', fontweight='bold')
    ax2.axis('off')

    ax3.imshow(G_true, cmap='Greens', vmax=1, vmin=0)
    ax3.set_title('"True" Green', fontweight='bold')
    ax3.axis('off')

    ax4.imshow(B, cmap='Blues', vmax=1, vmin=0)
    ax4.set_title('Blue', fontweight='bold')
    ax4.axis('off')

    plt.subplots_adjust(wspace=.02)
    plt.savefig('./temp/clipped.png')
    plt.clf()

    ######################################################################
    # The addition of the three channels results in a color image. Combine the three
    # channels with a stacked array and display the image with `imshow`.

    # The RGB array with the raw veggie band
    RGB_veggie = np.dstack([R, G, B])

    # The RGB array for the true color image
    RGB = np.dstack([R, G_true, B])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # The RGB using the raw veggie band
    ax1.imshow(RGB_veggie)
    ax1.set_title('GOES-16 RGB Raw Veggie', fontweight='bold', loc='left',
                fontsize=12)
    # ax1.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
    #             loc='right')
    ax1.axis('off')

    # The RGB for the true color image
    ax2.imshow(RGB)
    ax2.set_title('GOES-16 RGB True Color', fontweight='bold', loc='left',
                fontsize=12)
    # ax2.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
    #             loc='right')
    ax2.axis('off')

    plt.savefig('./temp/CombinedClipped.png')
    plt.clf()






    # Apply a gamma correction to the image to correct ABI detector brightness
    gamma = 2.2
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)
     # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)  # apply limits again, just in case.

    fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(1, 4, figsize=(16, 3))

    ax1.imshow(R, cmap='Reds', vmax=1, vmin=0)
    ax1.set_title('Red', fontweight='bold')
    ax1.axis('off')

    ax2.imshow(G, cmap='Greens', vmax=1, vmin=0)
    ax2.set_title('Veggie', fontweight='bold')
    ax2.axis('off')

    ax3.imshow(G_true, cmap='Greens', vmax=1, vmin=0)
    ax3.set_title('"True" Green', fontweight='bold')
    ax3.axis('off')

    ax4.imshow(B, cmap='Blues', vmax=1, vmin=0)
    ax4.set_title('Blue', fontweight='bold')
    ax4.axis('off')

    plt.subplots_adjust(wspace=.02)
    plt.savefig('./temp/gammaCorrected.png')
    plt.clf()


    ######################################################################
    # The addition of the three channels results in a color image. Combine the three
    # channels with a stacked array and display the image with `imshow`.

    # The RGB array with the raw veggie band
    RGB_veggie = np.dstack([R, G, B])

    # The RGB array for the true color image
    RGB = np.dstack([R, G_true, B])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # The RGB using the raw veggie band
    ax1.imshow(RGB_veggie)
    ax1.set_title('GOES-16 RGB Raw Veggie', fontweight='bold', loc='left',
                fontsize=12)
    # ax1.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
    #             loc='right')
    ax1.axis('off')

    # The RGB for the true color image
    ax2.imshow(RGB)
    ax2.set_title('GOES-16 RGB True Color', fontweight='bold', loc='left',
                fontsize=12)
    # ax2.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
    #             loc='right')
    ax2.axis('off')

    plt.savefig('./temp/CombinedGammaCorrected.png')

def convertData(filename):
    nc = Dataset(filename, 'r')
    height = 100
    width = 150
    upperLeftRow = 290
    upperLeftCol = 420
    R = nc["CMI_C02"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    G = nc["CMI_C03"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    B = nc ["CMI_C01"][upperLeftRow:upperLeftRow+height, upperLeftCol:upperLeftCol + width]
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    nc.close()
    nc = None 

    return np.dstack([R, G, B])

def getCloudData(start,numDays):
    # Use the anonymous credentials to access public data
    fs = s3fs.S3FileSystem(anon=True)
    #loop over every desired day
    for i in range(numDays):
        currentDay = start + datetime.timedelta(days=i)

        currentDayData = list()
        notFailed = True
        print('*********************starting day: ', i, ' of : ', numDays)
        #loop desired hours
        for ihr in range(8,19):
        # for ihr in range(24):
            date = currentDay + datetime.timedelta(hours=ihr)

            utcDate = date.astimezone(timezone('UTC'))
            text = 'noaa-goes16/ABI-L2-MCMIPC/{0}/{1}/{2}/'.format(utcDate.year,
                                                                str(utcDate.timetuple().tm_yday).zfill(3),
                                                                str(utcDate.hour).zfill(2))
            print(text)
            files = fs.ls(text)
            # print('files: ', files)
            if not files:
                notFailed = False
            if notFailed:

                # loop over each 5 minute interval in the hour
                currentHourData = list()
                # print(files[0])
                for x in files:
                    # print('hour: ', ihr)
                    # print("FILE: ", x)
                    filename = 'data/cloud/nc/'.format(year) + x.split('/')[-1]
                    fs.get(x, filename)
                    temp = convertData(filename)
                    # print('This Hours data: ', temp)
                    currentHourData.append(temp)
                    # print('CurrentHourData: ', np.asarray(currentHourData).shape)
                    # print('currentHourDataset shape: ', np.asarray(currentHourData).shape)
                    os.remove(filename)
                    # break

                currentDayData.append(currentHourData)
            # break
        if notFailed:

            print('*********************finished Day: ', i, ' of : ', numDays)
            toSaveFilename = './data/cloud/{0}/PartialDays/{1}.pickle'.format(start.year,
                                                                    str(currentDay.timetuple().tm_yday).zfill(3))
            dataToSave = np.asarray(currentDayData)
            print("shape of data: ", dataToSave.shape)
            pickle.dump(currentDayData,open( toSaveFilename, "wb" ))
        else:
            print('*********************finished Day: ', i, ' of : ', numDays, 'But something went wrong')
        # break

if __name__ == "__main__":

#change this to the month, day, and year that you want to start getting data for
#then get put the number of days that you want to get. This will fail if it crosses to another year. 
    year = 2018
    month = 5
    day = 1
    numDays = 120

    mountain = timezone('US/Mountain')
    start = datetime.datetime(year,month,day,00,tzinfo=mountain)
    getCloudData(start,numDays)
    # filename = './data/cloud/nc/OR_ABI-L2-MCMIPC-M3_G16_s20180021902201_e20180021904574_c20180021905088.nc'
    # displayData(filename)








#This produces a file names as such: 
#OR_<sensor>-<level>-<product short name>-M<scanning mode>-C<channel>-G<GOES Satellite>-s<start time>_e<end time>_c<central time>.nc

#examples of file system with explanations of ABI-L2-MCMIPC
#http://edc.occ-data.org/goes16/getdata/
