import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pickle
import tensorflow.keras as keras
import ClearSkyInput
import helperFunctions

font = {'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)


def plotAnimation(x1,x2,y,yPred,filename):


    plotx2 = np.zeros((72,1))
    plotx2[7:18] = x2[:11]
    plotx2[7+24:18+24] = x2[11:22]
    plotx2[7+48:18+48] = x2[22:]

    ploty = np.zeros((72,1))
    ploty[7:18] = y[:11]
    ploty[7+24:18+24] = y[11:22]
    ploty[7+48:18+48] = y[22:]

    plotyPred = np.zeros((72,1))
    plotyPred[7:18] = yPred[:11]
    plotyPred[7+24:18+24] = yPred[11:22]
    plotyPred[7+48:18+48] = yPred[22:]

    
    print('x1ToDisplay shape: ', x1.shape)
    print('plotx2 shape: ', plotx2.shape)
    print('ploty shape: ', ploty.shape)
    print('plotyPred shape: ', plotyPred.shape)

    hours = [ihr/24 for ihr in range(72)]

    
    
    #plot the cloudData to an imagea and save as multiple images or as an animation. 

    #create image with format (time,x,y)
    image = np.random.rand(72,10,10)
    image2 = np.random.rand(72,10,10)

    #setup figure
    fig = plt.figure()
    fig.tight_layout()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    #set up list of images for animation
    ims=[]
    for time in range(71):
        im = ax1.imshow(x1[time,6,:,:], animated=True)
        im2, = ax2.plot(plotx2[0:time],'g',animated=True)
        im3, = ax2.plot(ploty[0:time],'b', animated=True)
        im4, = ax2.plot(plotyPred[0:time],'r', animated=True)
        
        ims.append([im, im2,im3,im4])

    im = ax1.imshow(x1[71,6,:,:], animated=True)
    im2, = ax2.plot(plotx2[0:71],'g',animated=True, label='Clear Sky')
    im3, = ax2.plot(ploty[0:71],'b', animated=True, label='Actual')
    im4, = ax2.plot(plotyPred[0:71],'r', animated=True, label='Predicted')
    ax2.legend(loc = 'lower left',bbox_to_anchor=(0,.9))
    ims.append([im, im2,im3,im4])

    #run animation
    ani = anim.ArtistAnimation(fig,ims, interval=500,blit=True)
    
    ani.save(filename)

def plotPureResults(x2,y,yPred,filename):
    plt.clf()
    plotx2 = np.zeros((72,1))
    plotx2[7:18] = x2[:11]
    plotx2[7+24:18+24] = x2[11:22]
    plotx2[7+48:18+48] = x2[22:]

    ploty = np.zeros((72,1))
    ploty[7:18] = y[:11]
    ploty[7+24:18+24] = y[11:22]
    ploty[7+48:18+48] = y[22:]

    plotyPred = np.zeros((72,1))
    plotyPred[7:18] = yPred[:11]
    plotyPred[7+24:18+24] = yPred[11:22]
    plotyPred[7+48:18+48] = yPred[22:]

    
    fig = plt.figure()

    plt.plot(plotx2,'g',animated=True, label='Clear Sky')
    plt.plot(ploty,'b', animated=True, label='Actual')
    plt.plot(plotyPred,'r', animated=True, label='Predicted')
    plt.title('Results')
    fig.legend(loc = 'lower left',bbox_to_anchor=(.6,.6))
    plt.xlabel('Hours')
    plt.ylabel('Solar Energy')
    fig.tight_layout()
    plt.savefig(filename)
    
def scatterPlot(y,yPred,filename):
    plt.clf()
    # Create data
    x = y
    y = yPred
    colors = (0,0,0)
    area = np.pi*3

    # Plot
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.title('Predicted Vs Actual')
    plt.xlabel('Actual Solar Energy')
    plt.ylabel('Predicted Solar Energy')
    plt.tight_layout()
    plt.savefig(filename)




def testOnValidationDays():

    #Get validation X1
    file = open('./data/cloud/2018/March/day060.pickle','rb')
    day60 =  pickle.load(file, encoding='latin1')
    file.close()
    file = open('./data/cloud/2018/March/day061.pickle','rb')
    day61 =  pickle.load(file, encoding='latin1')
    file.close()
    file = open('./data/cloud/2018/March/day063.pickle','rb')
    day63 =  pickle.load(file, encoding='latin1')
    file.close()
    x1toDisplay = np.asarray(day60 + day61 + day63)

    x1 = np.asarray(day60[7:18] + day61[7:18] + day63[7:18])

    year = 2018
    days = (60,61,63)
    #get validation x2
    x2 = ClearSkyInput.getClearSkyIrradianceVariableDays(41.752032, -111.793835,year,days)
    x2 = np.asarray(x2).reshape(-1,1)

    #get respective ground truth
    # days = (32)
    y = helperFunctions.getSolarGroundTruthVariableDays(days,year-2000,'data/solarEnergy.csv')

    print('x1 shape: ', x1.shape)
    print('x2 shape: ', x2.shape)
    print('y shape: ', y.shape)







    # model = keras.models.load_model('./model/SolarPredictionNetwork.h5')
    # yPred = model.predict([x1,x2])

    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # scatterPlot(y,yPred,filename)
    


    number_dense_nodes = 16
    model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    yPred = model.predict([x1,x2])
    # filename = './results/validation/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/validation/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    filename = './results/validation/justResults{0}.png'.format(number_dense_nodes)
    plotPureResults(x2,y,yPred,filename)
    keras.backend.clear_session()

    # number_dense_nodes = 8
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/validation/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/validation/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/validation/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()

    # number_dense_nodes = 12
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/validation/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/validation/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/validation/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()

    # number_dense_nodes = 32
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/validation/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/validation/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/validation/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()

def testOnTrainingDays():
    #Get training X1
    file = open('./data/cloud/2018/FullDays/004.pickle','rb')
    day60 =  pickle.load(file, encoding='latin1')
    file.close()
    file = open('./data/cloud/2018/FullDays/007.pickle','rb')
    day61 =  pickle.load(file, encoding='latin1')
    file.close()
    file = open('./data/cloud/2018/FullDays/008.pickle','rb')
    day63 =  pickle.load(file, encoding='latin1')
    file.close()
    x1toDisplay = np.asarray(day60 + day61 + day63)

    x1 = np.asarray(day60[7:18] + day61[7:18] + day63[7:18])

    year = 2018
    days = (4,7,8)
    #get validation x2
    x2 = ClearSkyInput.getClearSkyIrradianceVariableDays(41.752032, -111.793835,year,days)
    x2 = np.asarray(x2).reshape(-1,1)

    #get respective ground truth
    # days = (32)
    y = helperFunctions.getSolarGroundTruthVariableDays(days,year-2000,'data/solarEnergy.csv')

    print('x1 shape: ', x1.shape)
    print('x2 shape: ', x2.shape)
    print('y shape: ', y.shape)
    

    number_dense_nodes = 16
    model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    yPred = model.predict([x1,x2])
    # filename = './results/training/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/training/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    filename = './results/training/justResults{0}.png'.format(number_dense_nodes)
    plotPureResults(x2,y,yPred,filename)
    keras.backend.clear_session()

    # number_dense_nodes = 8
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/training/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/training/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/training/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()

    # number_dense_nodes = 12
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/training/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/training/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/training/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()

    # number_dense_nodes = 32
    # model = keras.models.load_model('./model/{0}/SolarPredictionNetwork.h5'.format(number_dense_nodes))
    # yPred = model.predict([x1,x2])
    # filename = './results/training/Results{0}.mp4'.format(number_dense_nodes)
    # plotAnimation(x1toDisplay,x2,y,yPred,filename)
    # filename = './results/training/Results{0}.png'.format(number_dense_nodes)
    # scatterPlot(y,yPred,filename)
    # filename = './results/training/justResults{0}.png'.format(number_dense_nodes)
    # plotPureResults(x2,y,yPred,filename)
    # keras.backend.clear_session()


if __name__ == "__main__":
    #get model
    testOnValidationDays()
    # testOnTrainingDays()



