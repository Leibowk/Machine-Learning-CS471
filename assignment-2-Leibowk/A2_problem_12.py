import numpy as np

def findBeta(X, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), y)

def computeCovMatrix(X,y):

    beta = findBeta(X, y)

    predic = X @ beta

    return np.dot(y.T - y.mean(), predic - predic.mean(axis=0)) / (y.shape[0]-1)

# load data from http://lib.stat.cmu.edu/datasets/boston
#X features of x
X = np.loadtxt("housing_X.txt")

#y is target
y = np.loadtxt("housing_y.txt")

print("X shape is: ", X.shape)
print("y shape is: ", y.shape)
# test = np.stack((X, y), axis=0)
covariance = computeCovMatrix(X,y)
print(covariance)
# Beta = findBeta(X,y)
# print(Beta)

