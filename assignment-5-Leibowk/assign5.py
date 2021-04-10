'''
Kyle Leibowitz
CS471 Machine Learning
Kameron Harris
Gitrepo: leibowk@wwu.edu
'''

from mnist import MNIST
import numpy as np
import math
import matplotlib.pyplot as plt

def getData():
    # load dataset
    # download files from http://yann.lecun.com/exdb/mnist/
    #install the package https://pypi.python.org/pypi/python-mnist
    
    mndata = MNIST('./')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    labels_train = labels_train.astype(float)
    labels_test = labels_test.astype(float)

    trainFilter = np.logical_or(labels_train==7, labels_train == 2)
    testFilter = np.logical_or(labels_test==7, labels_test == 2)

    X_train, y_train = X_train[trainFilter], labels_train[trainFilter]
    X_test, y_test = X_test[testFilter], labels_test[testFilter]

    #Setting up Ys these are arrays in which every 7 has been switched to a 1, and every 2 to a -1
    y_train[y_train==7]=1
    y_train[y_train==2]=-1
    y_test[y_test==7]=1
    y_test[y_test==2]=-1

    return X_train, X_test, y_train, y_test

def phi(z):
    return 1.0/(1+np.exp(-z))

def gradResW(X, y, w, b, pen):
    return -np.mean(phi(-y*(b+X@w))*y*X.T, axis=1) + 2*pen*w


def gradResB(X, y, w, b):
    return -np.mean(phi(-y*(b+X@w))*y)

def costFun(X, y, w, b, pen):
    n, _ = X.shape
    cost = np.mean(np.log(1+np.exp(-y*(b+X@w)))) + pen*np.linalg.norm(w)**2
    return cost

def predFun(X, b, w):
    return np.sign(b+X@w)

def errorRate(yTrue, yPred):
    return (len(np.where(yTrue!=yPred)[0])/len(yTrue))*100

def getBatchParams(X, y, batchSize):
    n, _ = X.shape
    rows = np.random.choice(n, batchSize)
    return X[rows, :], y[rows]

def plotGradDescentFor2(X_train, y_train, X_test, y_test, h, stoch, batchSize):
    pen = .1
    n, d = X_train.shape
    tol = 1/math.sqrt(n)

    w= np.zeros(d)
    b= 0.0

    costTrain = []
    costTest = []
    errorTest = []
    errorTrain = []
    itter = []

    for i in range(1200):
        #Saving our old ones
        w_old = np.copy(w)
        b_old = b

        #Recalculating gradients and new paramaters
        #If we are doing SGD we will use our batch size to define our new x and b.
        if(stoch):
            xBatch, yBatch = getBatchParams(X_train, y_train, batchSize)
            w = w_old - h*gradResW(xBatch, yBatch, w_old, b_old, pen)
            b = b_old - h*gradResB(xBatch, yBatch, w_old, b_old)
        
        else:
            w = w_old - h*gradResW(X_train, y_train, w_old, b_old, pen)
            b = b_old - h*gradResB(X_train, y_train, w_old, b_old)

        #Saving costs. Only using what we've calculated using training data.
        cost = costFun(X_train, y_train, w, b, pen)
        costTrain.append(cost)
        testsCost = costFun(X_test, y_test, w, b, pen)
        costTest.append(testsCost)

        #Calculating the error rate using our predictions
        predTrain = predFun(X_train, b, w)
        predTest = predFun(X_test, b, w)
        errorTest.append(errorRate(y_test, predTest))
        errorTrain.append(errorRate(y_train, predTrain))
        
        itter.append(i)

        #Can use either convergence. Still not working.
        # converg = np.max(np.abs(w_old-w))
        # converg = np.abs((costFun(X_train, y_train, w, b, pen)-costFun(X_train, y_train, w_old, b_old, pen))/costFun(X_train, y_train, w_old, b_old, pen))
        # if(converg <= tol):
        #     break

    plt.plot(itter, costTrain)
    plt.plot(itter, costTest)
    plt.title("Gradient Descent Cost", fontsize=12) 
    plt.xlabel('Itterator')
    plt.ylabel('Cost')
    plt.legend(["Training Cost", "Testing Cost"])
    plt.show()

    plt.plot(itter, errorTrain)
    plt.plot(itter, errorTest)
    plt.title("Gradient Descent Error Percentage", fontsize=12) 
    plt.xlabel('Itterator')
    plt.ylabel('Error Rate Percentage')
    plt.legend(["Training Error", "Testing Error"])
    plt.show()


def getErrorFor3(X_train, y_train, X_test, y_test):
    # print("Shape X train: ", X_train.shape)
    # print("Shape y train: ", y_train.shape)
    # print("Shape X test: ", X_test.shape)
    # print("Shape y train: ", y_test.shape)
    
    it = 0
    pen = .1
    h = .01
    n, d = X_train.shape

    w = np.zeros(d)
    b= 0

    finalError = 0
    finalErrorTest = 0
    for i in range(1000):
        # print(i)
        #Saving our old ones
        w_old = np.copy(w)
        b_old = b
    
        w = w_old - h*gradResW(X_train, y_train, w_old, b_old, pen)
        b = b_old - h*gradResB(X_train, y_train, w_old, b_old)

        #Calculating the error rate using our predictions
        predTrain = predFun(X_train, b, w)
        error = errorRate(y_train, predTrain)
        predTest = predFun(X_test, b, w)
        errorTest = errorRate(y_test, predTest)
        # print(errorTest)

        if((finalError>error)|(finalError==0)):
            finalError=error
        
        if((finalErrorTest>errorTest)|(finalErrorTest==0)):
            finalErrorTest=errorTest
  

    return finalError, finalErrorTest

def getH(Xtrain, Xtest, sig):
    p= 3000
    _, d = Xtrain.shape
    G = np.random.randn(p, d) * sig**2
    c = np.random.rand(p) * 2*math.pi
    H_train = np.cos(Xtrain @ G.T + c)
    H_test = np.cos(Xtest @ G.T + c)
    return H_train, H_test

def getFitNValid(X, y):
    n,_ = X.shape
    rowsFit = math.floor(.8*n)
    return X[:rowsFit, :], X[rowsFit:, :], y[:rowsFit], y[rowsFit:]

def getSmallestSig(sigs, errors):
    smallest = np.argmin(errors)
    return sigs[smallest]

def part3Graph():
    X_train, X_test, y_train, y_test = getData()
    xFit, xValid, yFit, yValid = getFitNValid(X_train, y_train)

    sig = np.logspace(-3, 2)
    plt.xlim(10**-3, 10**2)
    plt.xscale('log')
    sigmas = []
    trainErrors = []
    testErrors = []
    i = 1

    #     hTrain, hTest = getH(X_train, X_test, trainSig)
    #     errorTrain, errorTest = getErrorFor3(hTrain, y_train, hTest, y_test)
    #     print("Official: ", errorTest)
    trainSig =  0.28117686979742307

    for elem in (sig):
        # hTrain, hTest = getH(X_train, X_test, trainSig)
        # errorTrain, errorTest = getErrorFor3(hTrain, y_train, hTest, y_test)
        # print("Official: ", errorTest)
        print(i)
        print("elem is: ", elem)
        hTrain, hValid= getH(xFit, xValid, elem)
        # hTrain, hTest = getH(X_train, X_test, elem)
        errorTrain, errorTest = getErrorFor3(hTrain, yFit, hValid, yValid)
        # errorTrain, errorTest = getErrorFor3(hTrain, y_train, hTest, y_test)
        # print("Training Error: ", errorTrain)
        # print("Testing Error: ", errorTest)
        # errorTrain, errorTest = getErrorFor3(hTrain, yFit, hValid, yValid)
        # errorTrain, errorTest = getErrorFor3(hTrain, y_train, hTest, y_test)
        sigmas.append(elem)
        trainErrors.append(errorTrain)
        print("All train: ", trainErrors)
        testErrors.append(errorTest)
        print("All test: ", testErrors)
        i = i + 1


    print("Smallest from train is: ", getSmallestSig(sigmas, trainErrors))
    print("Smallest from test is: ", getSmallestSig(sigmas, testErrors))
    plt.plot(sigmas, trainErrors)
    plt.plot(sigmas, testErrors)
    plt.title("Gradient Descent Error by Sigma", fontsize=12) 
    plt.xlabel('Sigma')
    plt.ylabel('Error Rate Percentage')
    plt.legend(["Train Error", "Validation Error"])
    plt.show()



##############################
########||| MAIN |||##########

#Problem 2
X_train, X_test, y_train, y_test = getData()
# plotGradDescentFor2(X_train, y_train, X_test, y_test, .1, False, None)
# plotGradDescentFor2(X_train, y_train, X_test, y_test, .01, True, 1)
# plotGradDescentFor2(X_train, y_train, X_test, y_test, .01, True, 100)


#Problem 3
# part3Graph()

#3.2
sig =   0.4498432668969444

##3.3
hTrain, hTest = getH(X_train, X_test, sig)
errorTrain, errorTest = getErrorFor3(hTrain, y_train, hTest, y_test)
print("Official: ", errorTest)