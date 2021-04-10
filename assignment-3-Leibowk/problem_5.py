import numpy as np
from pandas import read_csv

def augmX(X):
    n = X.shape[0]
    oneCol = np.ones(n)
    augX = np.concatenate((oneCol[:,None], X),axis=1)
    return augX

def MSE(trueY, predY):
    MSE = np.square(np.subtract(trueY, predY)).mean()
    return MSE

# 5.3
def ridge(X, alpha, y):
    '''
    Ridge regression function
    
    Parameters:
    X (np.ndarray, ndim=2): n x d data matrix of covariates (without augmenting using 1s)
    alpha (float): a positive float, the ridge parameter

    Returns:
    beta (np.ndarray, ndim=1): (1 + d) vector, first coefficient is intercept
    '''
    assert alpha > 0, 'alpha should be greater than zero'
    pass

    n, d = X.shape
    # form the augmented matrix X with a first column of 1s
    augX = augmX(X)

    # form the D matrix
    D = np.full(shape=d+1, fill_value=1, dtype=np.float)
    D[0] = 0
    D = np.diag(D)
    
    # solve the equations
    #  (X^T*X+alpha*D)Beta = X^T*y
    Beta = np.linalg.solve(((augX.T @ augX)+alpha * D),(augX.T @ y))
    return Beta

# Pandas is a useful library for working with data
data = read_csv("prostate.data", sep='\t')

# We will convert it into a np.ndarray
# we will predict lpsa (log prostate specific antigen)
target = 'lpsa'
# based on the following features
features = ['lcavol', 'lweight', 'age', 'lbph',
            'svi', 'lcp', 'gleason', 'pgg45']
# split data into training/test sets based on the train flag
is_train = data.train == 'T'
X, y = data[features].values, data[target].values
X_train, y_train = X[is_train], y[is_train]
X_test, y_test = X[~is_train], y[~is_train]

# 5.1 estimate the means and standard deviations of the features using just the training data
mean_vec = np.mean(X_train, axis=0)
print("mean_vec: ", mean_vec, "\n")
std_vec = np.std(X_train, axis=0)
print("std_vec: ", std_vec, "\n")

# 5.2 standardize X_train and X_test and form matrices X_train_std, X_test_std
#lesson 11, 3rd slide. are formulas
X_train_std = np.divide(np.subtract(X_train, mean_vec), std_vec)
X_test_std = np.divide(np.subtract(X_test, mean_vec), std_vec)

# make sure the following lines are included afterwards to save your results
# np.savetxt("X_train_std.txt", X_train_std)
# np.savetxt("X_test_std.txt", X_test_std)

# 5.4
#
#
for test in [1, .1, 10]:
    Beta = ridge(X_train_std, test, y_train)
    ypred = augmX(X_train_std) @ Beta
    print("MSE of B-Ridge Train @ ",test, " ", MSE(y_train, ypred))

    Beta = ridge(X_test_std, test, y_test)
    ypred = augmX(X_test_std) @ Beta
    print("MSE of B-Ridge Test @ ",test, " ", MSE(y_test, ypred))
    print(" ")
