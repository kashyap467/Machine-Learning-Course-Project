import scipy.io         # to load .mat file
import numpy as np      # to perform mathematical operations on dataset
import matplotlib.pyplot as plt

import GaussianNaiveBayes as NB, LogisticRegression as LR

if __name__ == '__main__': 

    ''' LOADING DATASET '''
    # Read from .mat file and load as training and testing data separately
    fm_data = scipy.io.loadmat('./fashion_mnist.mat')
    trX, trY = fm_data['trX'], fm_data['trY'].T
    tsX, tsY = fm_data['tsX'], fm_data['tsY'].T


    ''' PREPARING TRAINING DATA '''
    # Get Mean, StdDev features of each sample in Training Set and use them as input values
    trX = np.array([[trX[i].mean(), trX[i].std()] for i in range(trX.shape[0])])
    trData = np.concatenate((trX, trY), axis=1)     # Data -> [Mean   StdDev   Label]

    plt.scatter(trX[:,0], trX[:,1])
    plt.title("Scatter plot of given samples")
    plt.show()

    # Split training data based on Class labels '0' & '1'
    trData0 = trData[trData[:,2] == 0]
    trData1 = trData[trData[:,2] == 1]

    # Calculate prior probabilities, p(y=0) and p(y=1) for training data
    countTrY0 = len(trData[trData[:,2] == 0])
    prior_tr_0 = countTrY0 / len(trY)

    ''' PREPARING TESTING DATA '''
    # Get Mean, StdDev features of each sample in Testing Set and use them as input values
    tsX = np.array([[tsX[i].mean(), tsX[i].std()] for i in range(tsX.shape[0])])
    tsData = np.concatenate((tsX, tsY), axis=1)     # Data -> [Mean   StdDev   Label]

    # Split testing data based on Class labels '0' & '1'
    tsData0 = tsData[tsData[:,2] == 0]
    tsData1 = tsData[tsData[:,2] == 1]

    sampleCntY0 = tsData0.shape[0]
    sampleCntY1 = tsData1.shape[0]



    ''' METHOD TO CALCULATE ACCURACY '''
    def get_accuracy (predCnt_0, predCnt_1) :
        acc_0 = ( predCnt_0 / sampleCntY0 ) * 100       # accuracy for '0'
        print("Accuracy for Class '0' : {} %".format(round(acc_0, 2)))

        acc_1 = ( predCnt_1 / sampleCntY1 ) * 100       # accuracy for '1'
        print("Accuracy for Class '1' : {} %".format(round(acc_1, 2)))

        acc_total = ( (predCnt_0 + predCnt_1) / (sampleCntY0 + sampleCntY1) ) * 100   # total accuracy of LR
        return acc_total



    '''  
GAUSSIAN NAIVE BAYES  
    '''
    print('\nGaussian Naive Bayes on Fashion MNIST Dataset\n'+'-'*50)

    # Get estimated parameters by performing Maximum Likelihood Estimation
    params = NB.estimate_params(trData)
    # Get prediction values for both labels
    predCnt_0, predCnt_1 = NB.NaiveBayes(tsData, params, prior_tr_0)

    # Get Accuracy for Naive Bayes
    acc_total = get_accuracy(predCnt_0, predCnt_1)
    print('Accuracy using Naive Bayes : {} %\n'.format(round(acc_total, 2)))



    '''  
LOGISTIC REGRESSION  
    '''
    print('\nLogistic Regression on Fashion MNIST Dataset\n'+'-'*50)

    # Input the larning rate and number of itereations for Gradient Ascent to run.
    learn_rate = 0.1 #0.1
    num_epoch = 1000 #1000
    coeffs = LR.estimate_coeffs(trData, trY, learn_rate, num_epoch)
    # Get prediction values for both labels
    predCnt_0, predCnt_1 = LR.LogisticRegression(tsData, coeffs)

    # Get Accuracy for Logistic Regression
    acc_total = get_accuracy(predCnt_0, predCnt_1)
    print('With epochs = %d  &  Learn Rate = %.2f, ' % (num_epoch, learn_rate))
    print('Accuracy using Logistic Regression : {} % \n'.format(round(acc_total, 2)))
