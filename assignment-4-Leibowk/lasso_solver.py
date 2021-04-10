import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lasso_solver(X, y, beta_init, penalty, tol):
    '''
    Solve the Lasso equations using coordinate descent.

    Parameters:
    X (np.ndarray, ndim=2): n x d data matrix without extra column of 1s
    y (np.ndarray, ndim=1): n x 1 label vector
    beta_init (np.ndarray, ndim=1): (d + 1) x 1 initial guess
    penalty (float): Lasso penalty >= 0, lambda in our math
    tol (float): convergence tolerance >= 0, delta in our math 

    Returns:
    beta (np.ndarray, ndim=1): (d + 1) x 1 solution
    '''
    assert penalty >= 0, "penalty should be >= 0"
    assert tol >= 0, "tolerance should be >= 0"
    n, d = X.shape
  
    beta = np.copy(beta_init)
    beta_old = np.copy(beta)

    #precomputing for efficiency 
    a = 2*np.sum(X**2, axis=0)

    converged = False  
    while not converged:
        beta_zero = (1/n) * np.sum(y - X @ beta_old)

        for i in range(d):
            bi = np.sum(X[:,i]*(y-beta_zero-np.delete(X,i,axis=1)@np.delete(beta,i,axis=0)))*2
            beta[i] = softThresh(penalty, bi)/ a[i]
         
        beta_old = np.copy(beta)
        converg = np.max(np.abs(beta-beta_old))
        if(converg <= tol):
            converged = True
        if converged:
            break

    return beta

def wiCalc(j,k):
    '''
    Helper function for calcW
    '''
    if(k>=j):
        return j/k
    return 0

def calcW(d, k):
    '''
    Returns a np array of size d that either has j/k or zeroes
    '''
    w = []
    for i in range(d):
        w.append(wiCalc(i, k))
    return np.array(w)

def softThresh(pen, z):
    if(z<-pen):
        return z+pen
    if(z>pen):
        return z-pen
    return 0

def lamMax(y, X):
    return np.amax(2*np.abs(X.T @ (y-  np.mean(y))))

def TPR(beta, k):
    num=0
    for i in range(k):
        if(beta[i]!=0):
            num=num+1
    return num/k

def FDR(beta, k, d):
    num=0
    for i in range(k+1, d):
        if(beta[i]!=0):
            num=num+1
    norm = np.linalg.norm(beta, ord=0, axis=0)
    if(norm!=0):
        return num/np.linalg.norm(beta, ord=0, axis=0)
    return 0

def MSE(trueY, predY):
    aMSE = np.square(np.subtract(trueY, predY)).mean()
    return aMSE


def prob4():
    '''
    Testing on real data for part 4

    '''

    # Import data
    df_train = pd.read_table("crime-train.txt").to_numpy()
    df_test = pd.read_table("crime-test.txt").to_numpy()

    yTrain = df_train[:,0]
    yTest = df_test[:,0]
    xTrain = df_train[:,1:]
    xTest = df_test[:,1:]
    
    # set up bounds
    n, d = np.shape(xTrain)
    
    tol = .0001

    xTrain = np.insert(xTrain, 0, 1, axis=1)
    xTest = np.insert(xTest, 0, 1, axis=1)
    labmda_Max = lamMax(yTrain, xTrain)
    beta_init = np.zeros(d+1)
    
    ratio = 1

    #Making all lists that will hold all points for problems.
    #3.1
    zNorms = []
    lambdas = []

    #3.2
    agePct12t29 = []
    pctWSocSec = []
    pctUrban = []
    agePct65up = []
    householdsize = []

    #3.3
    trainMSE = []
    testMSE = []
    
    # 4.1 Plot the number of non zeros
    while True:
        penalty = labmda_Max / ratio
        
        #beta = beta of training data
        beta = lasso_solver(xTrain, yTrain, beta_init, penalty, tol)
        #betaTest = beta of testing data
        betaTest = lasso_solver(xTest, yTest, beta_init, penalty, tol)

        current_Z_norm = np.linalg.norm(beta, ord=0, axis=0)
        zNorms.append(current_Z_norm)
        lambdas.append(penalty)

        # Positions of agePct12t29, pctWSocSec, pctUrban, agePct65up , householdsize
        #5, 14, 9, 7, 3
        agePct12t29.append(beta[5])
        pctWSocSec.append(beta[14])
        pctUrban.append(beta[9])
        agePct65up.append(beta[7])
        householdsize.append(beta[3])

        #ypred = prediction of y for training
        ypred = xTrain @ beta
        #yTestPred = prediction of y for test
        yTestPred = xTest @ beta

        currTrainMSE = MSE(yTrain, ypred)
        trainMSE.append(currTrainMSE)
        currTestMSE = MSE(yTest, yTestPred)
        testMSE.append(currTestMSE)

        ratio = ratio * 2

        beta_init = np.copy(beta)
        if(.01 > penalty):
            break
    
    #4.4 Work
    # beta = lasso_solver(xTrain, yTrain, beta_init, 30, tol)
    # print(beta)
    # print(np.argmax(beta))
    # print(np.argmin(beta))

    # Plot the nonzeros vs lambda 4.1
    plt.plot(lambdas, zNorms)
    plt.xscale("log")
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non-Zeros')
    plt.title("Number of Non-Zeros vs Penalty", fontsize=12) 
    plt.show()

    # plot agePct12t29, pctWSocSec, pctUrban, agePct65up , householdsize Betas vs lambda 4.2   
    plt.plot(lambdas, agePct12t29)
    plt.plot(lambdas, pctWSocSec)
    plt.plot(lambdas, pctUrban)
    plt.plot(lambdas, agePct65up)
    plt.plot(lambdas, householdsize)
    plt.xscale("log")
    plt.xlabel('Lambda')
    plt.ylabel('Values of coefficients in B')
    plt.title("Specific Beta Coefficients vs Penalty", fontsize=12) 
    plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])
    plt.show()

    # plot MSE versus lambda
    plt.plot(lambdas, trainMSE)
    plt.plot(lambdas, testMSE)
    plt.xscale("log")
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title("MSE Plotted against Penalty", fontsize=12) 
    plt.legend(["train", "test"])
    plt.show()


def prob3():
    '''
    Testing on fake data for part 3
    '''
    n = 500
    d = 1000
    k = 100
    sigma = 1
    tol = .0001

    w = calcW(d, k)
    epsi = np.random.randn(n) * sigma
    X = X = np.random.randn(n,d)
    y = X @ w + epsi
    X = np.insert(X, 0, 1, axis=1)

    beta_init = np.zeros(d+1)
    lambMax = lamMax(y, X)
    ratio = 0

    cords = []
    lambdas = []
    FDRs = []
    TPRs = []

    while True:
        penalty = lambMax/ (1.5**ratio)
        beta = lasso_solver(X, y, beta_init, penalty, tol)

        currZNorm = np.linalg.norm(beta, ord=0, axis=0)
        
        cords.append(currZNorm)
        lambdas.append(penalty)
        FDRs.append(FDR(beta, k, d))
        TPRs.append(TPR(beta, k))

        ratio = ratio + 1
        if(currZNorm >=.99*d):
            break

    plt.plot(lambdas, cords)
    plt.title("Synthetic Data Num Nonzeroes vs Lambda", fontsize=12)  
    plt.xscale("log")
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non Zeroes')
    plt.show()

    FDRs = np.array(FDRs)
    plt.scatter(FDRs, TPRs, c=np.arange(FDRs.shape[0]), cmap="viridis")
    plt.title("Synthetic Data FDR vs TPR", fontsize=12)  
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()



#MAIN
prob3()
prob4()