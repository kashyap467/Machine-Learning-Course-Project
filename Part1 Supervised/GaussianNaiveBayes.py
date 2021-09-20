import numpy as np      # to perform mathematical operations on dataset
import matplotlib.pyplot as plt
from scipy.stats import norm


def CalculatePDF(MeanStd, x):
    ''' Calculate the Probability Density Function (PDF) with given mean, stdev and feature values '''
    consTerm = 1 / (np.sqrt(2*np.pi) * MeanStd[1])
    expoTerm = np.exp(-0.5*(((x - MeanStd[0]) / MeanStd[1])**2))
    return (consTerm * expoTerm)


'''
PARAMETER ESTIMATION USING MAXIMUM LIKELIHOOD ESTIMATION
'''
def estimate_params(trData) : 
    '''
    Estimate parameters 'Sample Mean' & 'Sample Standard Deviation' 
        for the two features (mean, std) and two classes '0', '1'    ->   total 8 params.
    '''
    print('Estimated Parameters : ')

    # Split training data based on Class labels '0' & '1'
    trData0 = trData[trData[:,2] == 0]
    trData1 = trData[trData[:,2] == 1]

    # Find mean, std of means and standard deviations from training samples in Y=0
    meanMu0 , stdMu0  =  trData0[:,0].mean() , trData0[:,0].std()
    maxMu0 , minMu0 = trData0[:,0].max(), trData0[:,0].min()
    meanSd0 , stdSd0  =  trData0[:,1].mean() , trData0[:,1].std()
    maxSd0 , minSd0 = trData0[:,1].max(), trData0[:,1].min()
    print("Class '0' : ")
    print("  Feature 1 -  Mean : {0} \tStDev : {1} \n\t\tMax : {2} \t  Min : {3}".format(meanMu0, stdMu0, maxMu0, minMu0))
    print("  Feature 2 -  Mean : {0} \tStDev : {1} \n\t\tMax : {2} \t  Min : {3}".format(meanSd0, stdSd0, maxSd0, minSd0))
    
    # Create parameters         ->  | Mean(means_0), Std(means_0) |
    #  matrix for class '0'         | Mean(stdvs_0), Std(stdvs_0) |
    paramsY0 = np.array([meanMu0, stdMu0, meanSd0, stdSd0]).reshape(2,2)


    # Find mean, std of means and standard deviations from training samples in Y=1
    meanMu1 , stdMu1  =  trData1[:,0].mean() , trData1[:,0].std()
    maxMu1 , minMu1 = trData1[:,0].max(), trData1[:,0].min()
    meanSd1 , stdSd1  =  trData1[:,1].mean() , trData1[:,1].std()
    maxSd1 , minSd1 = trData1[:,1].max(), trData1[:,1].min()
    print("Class '1' : ")
    print("  Feature 1 -  Mean : {0} \tStDev : {1} \n\t\tMax : {2} \t  Min : {3}".format(meanMu1, stdMu1, maxMu1, minMu1))
    print("  Feature 2 -  Mean : {0} \tStDev : {1} \n\t\tMax : {2} \t  Min : {3}\n".format(meanSd1, stdSd1, maxSd1, minSd1))

    # Create parameters         ->  | Mean(means_1), Std(means_1) |
    #  matrix for class '1'         | Mean(stdvs_1), Std(stdvs_1) |
    paramsY1 = np.array([meanMu1, stdMu1, meanSd1, stdSd1]).reshape(2,2)


    fig, axs = plt.subplots(2)#, figsize=(10,10))

    x_Mu0 = np.linspace(meanMu0 - 3*stdMu0, meanMu0 + 3*stdMu0, 100)
    # x_Mu0 = np.arange(meanMu0 - 3*stdMu0, meanMu0 + 3*stdMu0, 0.01)
    # x_Mu0 = np.arange(minMu0, maxMu0, 0.01)

    x_Sd0 = np.linspace(meanSd0 - 3*stdSd0, meanSd0 + 3*stdSd0, 100)
    # x_Sd0 = np.arange(meanSd0 - 3*stdSd0, meanSd0 + 3*stdSd0, 0.01)
    # x_Sd0 = np.arange(minSd0, maxSd0, 0.01)
    axs[0].plot(x_Mu0, norm.pdf(x_Mu0, meanMu0, stdMu0), label='Feature 1')
    axs[0].vlines(x=meanMu0, ymin=0, ymax=norm.pdf(meanMu0, meanMu0, stdMu0), linestyles='dashed')
    axs[0].hlines(y=2.25, xmin=meanMu0, xmax=(meanMu0+stdMu0), linestyles='dashed')
    axs[0].plot(x_Sd0, norm.pdf(x_Sd0, meanSd0, stdSd0), label='Feature 2', color='orange')
    axs[0].vlines(x=meanSd0, ymin=0, ymax=norm.pdf(meanSd0, meanSd0, stdSd0) , linestyles='dashed', color='orange')
    axs[0].hlines(y=2.75, xmin=meanSd0, xmax=(meanSd0+stdSd0), linestyles='dashed', color='orange')
    axs[0].set_title("Class '0'")
    axs[0].grid()
    axs[0].set_xticklabels([])
    axs[0].legend(loc='upper right', shadow=True, fancybox=True)

    x_Mu1 = np.linspace(meanMu1 - 3*stdMu1, meanMu1 + 3*stdMu1, 100)
    # x_Mu1 = np.arange(meanMu1 - 3*stdMu1, meanMu1 + 3*stdMu1, 0.01)
    # x_Mu1 = np.arange(minMu1, maxMu1, 0.01)

    x_Sd1 = np.linspace(meanSd1 - 3*stdSd1, meanSd1 + 3*stdSd1, 100)
    # x_Sd1 = np.arange(meanSd1 - 3*stdSd1, meanSd1 + 3*stdSd1, 0.01)
    # x_Sd1 = np.arange(minSd1, maxSd1, 0.01)
    axs[1].plot(x_Mu1, norm.pdf(x_Mu1, meanMu1, stdMu1), label='Feature 1')
    axs[1].vlines(x=meanMu1, ymin=0, ymax=norm.pdf(meanMu1, meanMu1, stdMu1), linestyles='dashed')
    axs[1].hlines(y=4.25, xmin=meanMu1, xmax=(meanMu1+stdMu1), linestyles='dashed')
    axs[1].plot(x_Sd1, norm.pdf(x_Sd1, meanSd1, stdSd1), label='Feature 2', color='orange')
    axs[1].vlines(x=meanSd1, ymin=0, ymax=norm.pdf(meanSd1, meanSd1, stdSd1), linestyles='dashed', color='orange')
    axs[1].hlines(y=4.25, xmin=meanSd1, xmax=(meanSd1+stdSd1), linestyles='dashed', color='orange')
    axs[1].set_title("Class '1'")
    axs[1].grid()
    axs[1].legend(loc='upper right', shadow=True, fancybox=True)

    plt.show()


    # merge and return the whole parameters matrix
    return (np.concatenate((paramsY0, paramsY1), axis=0))


def NaiveBayes (tsData, params, prior_tr_0) :

    # Using Univariate Gaussian Dsitribution PDF formula (since data is assumed to be from Gaussian Distribution and 
    #  Naive Bayes assumes conditional independency of features.

    prior_tr_1 = 1 - prior_tr_0

    # Split testing data based on Class labels '0' & '1'
    tsData0 = tsData[tsData[:,2] == 0]
    tsData1 = tsData[tsData[:,2] == 1]

    # Intialize right prediction counters for classes '0' & '1'
    sampleCntY0 = tsData0.shape[0]
    rightPredY0Cnt = 0              # for CLass '0'
    sampleCntY1 = tsData1.shape[0]
    rightPredY1Cnt = 0              # for Class '1'


    ''' PREDICT CLASS (Y) VALUES FOR ALL TESTING SAMPLES UNDER Y=0 '''

    # Intialize 1000*1 array with zeros to store predicted Y values
    predY0 = np.zeros(shape = (sampleCntY0, 1))

    # Calculate p(y=0|x1,x2) & p(y=1|x1,x2) for all test samples under Y=0
    # for y=0
    pdfMus0 = CalculatePDF(params[0,:], tsData0[:,0])       # Likelihood of feature 1 for y=0
    pdfSds0 = CalculatePDF(params[1,:], tsData0[:,1])       # Likelihood of feature 2 for y=0
    probMuSdY0 = prior_tr_0 * (pdfMus0 * pdfSds0)
    # for y=1
    pdfMus1 = CalculatePDF(params[2,:], tsData0[:,0])       # Likelihood of feature 1 for y=1
    pdfSds1 = CalculatePDF(params[3,:], tsData0[:,1])       # Likelihood of feature 2 for y=1
    probMuSdY1 = prior_tr_1 * (pdfMus1 * pdfSds1)

    for i in range(sampleCntY0) :
        if probMuSdY1[i] > probMuSdY0[i] :    # Assign Predicted Label
            predY0[i] = 1
        if predY0[i] == tsData0[i,2] :        # for right prediction, update appropriate counter
            rightPredY0Cnt += 1

    
    ''' PREDICT CLASS (Y) VALUES FOR ALL TESTING SAMPLES UNDER Y=1 '''

    # Intialize 1000*1 array with ones to store predicted Y values
    predY1 = np.ones(shape = (sampleCntY1, 1))

    # Calculate p(y=0|x1,x2) & p(y=1|x1,x2) for all test samples under Y=1
    #  for y=0
    pdfMus0 = CalculatePDF(params[0,:], tsData1[:,0])       # Likelihood of feature 1 for y=0
    pdfSds0 = CalculatePDF(params[1,:], tsData1[:,1])       # Likelihood of feature 2 for y=0
    probMuSdY0 = prior_tr_0 * (pdfMus0 * pdfSds0)
    #  for y=1
    pdfMus1 = CalculatePDF(params[2,:], tsData1[:,0])       # Likelihood of feature 1 for y=1
    pdfSds1 = CalculatePDF(params[3,:], tsData1[:,1])       # Likelihood of feature 2 for y=0
    probMuSdY1 = prior_tr_1 * (pdfMus1 * pdfSds1)

    for i in range(sampleCntY1) :
        if probMuSdY0[i] > probMuSdY1[i] :    # Assign Predicted Label
            predY1[i] = 0
        if predY1[i] == tsData1[i,2] :        # for right prediction, update appropriate counter
            rightPredY1Cnt += 1
    
    return rightPredY0Cnt, rightPredY1Cnt