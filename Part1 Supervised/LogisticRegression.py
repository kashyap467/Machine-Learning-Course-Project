import numpy as np      # to perform mathematical operations on dataset


def sigmoid(x):
    ''' method for sigmoid function  ->   sigmoid(x) = 1 / (1 + e^(-x)) '''
    return 1.0 / (1.0 + np.exp(-x))


'''
PARAMETER ESTIMATION USING GRADIENT ASCENT METHOD
'''
def estimate_coeffs(trData, trY, learn_rate, num_epoch):
    '''
    Estimate coefficients using Gradient Ascent method with given learning rate & number of iterations
    '''
    # Create a matrix of type [1  x1  x2] to multiply with [w0  w1  w2] and get wTx
    ones = np.ones(shape = (trData.shape[0], 1))
    train = (np.concatenate((ones, trData), axis=1))[:,:-1]
    
    # Initiate coeff matrix [w0  w1  w2] with 0s
    coeffs = np.zeros(shape = (train.shape[1], 1))

    # Interate for 'num_epoch' times to estimate coeffs
    for epoch in range(num_epoch) :
        wTx = np.dot(train, coeffs)         # Find wTx = (w0 + w1.x1 + w2.x2) for whole training data
        sigmWtX = sigmoid(wTx)              # Get Z = sigmoid(wTx)
        error = np.subtract(trY, sigmWtX)   # Calculate Y-Z
        dervTerm = np.dot(train.T, error)   # Calculate dL(w)/dw = (Y-Z).X
        learnTerm = learn_rate * dervTerm   # Calculate learning term = n.(dL(w)/dw)  [n -> learning rate]
        coeffs = np.add(coeffs, learnTerm)  # Get updated W's => W' = W + n.(dL(w)/dw)

    print('Estimated Coefficients: \n W0 = {0},\t W1 = {1},\t W2 = {2}\n'.format(coeffs[0,0], coeffs[1,0], coeffs[2,0]))
    return coeffs       # return final estimated coefficients


def LogisticRegression (tsData, coeffs) : 

    ''' CALCULATE p(y|x) = sigmoid(wTx) FOR TEST DATA AND CLASSIFY '''

    # Initialize right prediction counters for classes '0','1', and total accuracy
    rightLRPredCnt = rightLRPred0Cnt = rightLRPred1Cnt = 0

    # Create [1  x1  x2] type matrix from test data to multiply with estimated coeffs [w0  w1  w2]
    ones = np.ones(shape = (tsData.shape[0], 1))
    test = (np.concatenate((ones, tsData), axis=1))[:,:-1]

    # Calculate p(y|x1,x2) and classify test data
    probTsY = sigmoid(np.dot(test, coeffs))
    predY = np.around(probTsY)

    for i in range(predY.shape[0]) :
        if predY[i][0] == tsData[i,2] :
            rightLRPredCnt += 1          # if right prediction, update counter
            if predY[i][0] == 1 :        # for right prediction of class '1', update its counter
                rightLRPred1Cnt += 1
            else :                       # for right prediction of class '0', update its counter
                rightLRPred0Cnt += 1
    
    return rightLRPred0Cnt, rightLRPred1Cnt
    