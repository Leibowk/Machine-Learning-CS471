import numpy as np
import matplotlib.pyplot as plt

def poly_features(X, p):
    '''
    Compute the polynomial features of degree p

    Parameters:
    X (np.ndarray, ndim=2): n x d data matrix
    p (int): degree of polynomial

    Returns:
    F (np.ndarray, ndim=2): n x f feature matrix  

    DOES ASSUME OF POWER Greater than 1  
    
    May want to switch X[i][j]
    to 
    X[i,j]

    X[i,:] could be better
    '''


    numcols = len(X[0])
    numrows = len(X)

    #Building F
    F = []
    for i in range(numrows):
        # Building by row
        row = []

        # Assumes at least has power of zero or more
        row.append(1)

        #After that start inserting based on power (p)
        if p>=1:
            # Inserting all originals
            for y in range(numcols):
                row.append(X[i][y])
            
            deg=2
            # Inserting x^deg
            while(deg<=p):
                for y in range(numcols):
                    row.append(X[i][y]**deg)
                deg = deg + 1
        
        F.append(row)
   
    F = np.array(F)
    return F

# Function to build a matrix of the uniformly divided
# n_test number of points
def buildUniform(n_test):
    m= np.sqrt(n_test).astype(int)
    iterator = 1/m
    # print(iterator)
    # iterator = 1/n_test
    allCords = []
    currItX = 0
    while(currItX<1):
        currItY = 0
        while(currItY<1):
            currCord=[currItX, currItY]
            allCords.append(currCord)
            currItY = currItY + iterator
        currItX = currItX + iterator
    
    # plt.xlabel('x')
    # plt.ylabel('y')
   
    # print(allCords)
    # print("\n")
  
    # for i in range(len(allCords)):
    #     plt.plot(allCords[i][0], allCords[i][1], 'ro')

    # plt.show()

    return np.array(allCords)

def true_fun(X):
    '''
    The true target function
    '''
    return np.cos(1.5 * np.pi * X[:,0]) - 0.3 * np.sin(2 * np.pi * X[:,1])


def findBeta(X, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), y)

def test_error(X, y, n_test, numPolys):
    '''
    Compute the average testing squared error

    Write a function that evaluates the test error numerically. Generate a uniformly spaced
    grid of ntest points over [0, 1]2 = {~x ∈ R 2
    : 0 ≤ x1, x2 ≤ 1}. For each point in the test set, evaluate
    the true function and the estimate returned by polynomial regression (the polynomial function fit
    using the training set of nsamples generated in the prototype code), and compute the squared error.
    The function should return the mean square error (MSE) over all ntest points, the test MSE. See the
    function prototype provided. It is ok if it only works with ntest equal to a square.

    '''
    
    featuresX = poly_features(X, numPolys)
    # print("Feat shape is: ", featuresX.shape)

    Beta = findBeta(featuresX, y)
    # print("Beta shape is: ", Beta.shape)
    # print(Beta)

    # train error = Norm(XB-y)^2
    # X is features of X
    #XB is nx1 and it's the prediction for degree of polynomial
    XB = featuresX @ Beta
    difference = XB - y
    trainError =  np.linalg.norm(difference)**2
    # print(trainError)

    #G is grid
    #G' is Features of G
    #Test error = Norm(G'B-y)^2
    uniformArray = buildUniform(n_test)
    # print("Uniform Matrix shape is: ", uniformArray.shape)
    uniFeats = poly_features(uniformArray, numPolys)
    # print("Uniform Matrix feats shape is: ", uniFeats.shape)

    predic = uniFeats @ Beta
    print("(Features of Uni * Beta) shape is: ", predic.shape)

    # error = 0
    dif2 = predic - y

    mean_squared_error = error**2

    #mean_squared_error = squaredError...
    
    #poly features gives matrix. Plug matrix into equation that returns function of best fit line
    #B = (X^TX)-1X^Ty (Best guess)
    # X is feature matrix
    # y is the real output


    # For each point in the test set, evaluate the true function and the estimate returned by polynomial regression (the polynomial function fit using the training set of nsamples generated in the prototype code), and compute the squared error.

    return mean_squared_error





np.random.seed(0) # do not change this

n_samples = 40

X = np.random.rand(n_samples, 2)
# print("X shape is: ", X.shape)

y = true_fun(X) + np.random.randn(n_samples) * 0.2
# print("y shape is: ", y.shape)


# Fixing random state for reproducibility


test_error(X, y, 16, 2)