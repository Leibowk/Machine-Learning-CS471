import numpy as np
import matplotlib.pyplot as plt


def PCA(X):
    '''
    Compute the PCA of X = U S V^T

    Parameters:
    X (np.ndarray, ndim=2): n x d data matrix

    Returns:
    U (np.ndarray, ndim=2): n x d left singular vectors
    S (np.ndarray, ndim=1): d x d vector of singular values
    V (np.ndarray, ndim=2): d x n array of right singular vectors
    '''
    # step 1: subtract column means
    arrayOfMeans = np.outer(np.ones(n), np.mean(X, axis=0).T)

    X = np.subtract(X, arrayOfMeans)

    # step 2: take svd
    #return U, S, V
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U, S, V

def checkPCA(X):
    onesNcolMeans = np.outer(np.ones(n), np.mean(X, axis=0).T)
    U, S, V = PCA(X)
    
    S = np.diag(S)
    USV = U @ S @ V
    simMatr = USV + onesNcolMeans

    res = np.allclose(X, simMatr)
    if res:
        print("The arrays are similar!")
    else:
        print("Sorry, non similar arrays :(")

def frac_var_explained(s):
    '''
    Compute the fraction of variance explained

    Parameters:
    s (np.ndarray, ndim=1): p vector of singular values

    Returns:
    f (np.ndarray, ndim=1): p vector of cumulative variance explained
    '''
    size= s.shape
    totSumSquared = np.sum(np.square(s))
    fracOfVar = []

    sumSquared = 0
    for i in range(size[0]):
        sumSquared = sumSquared + s[i]**2
        fracOfVar.append(sumSquared/totSumSquared)
    
    fracOfVar = np.array(fracOfVar)
    return fracOfVar

def plotFracVarExpl(varExpl):
    plt.plot(varExpl)
    plt.plot(np.arange(60), np.full(shape=60, fill_value=.95, dtype=np.float))
    plt.xlabel('Number of components')
    plt.ylabel('Fraction of Variance Explained')
    plt.show()
    
def makeUXbar(X, U):
    arrayOfMeans = np.outer(np.ones(n), np.mean(X, axis=0).T)
    xBar = np.subtract(X, arrayOfMeans)

    newMatr = U[:,:2].T @ xBar
    return newMatr

def plotYvrsTime(newMatr):  
    plt.plot(newMatr[0,:])
    plt.plot(newMatr[1,:])
    plt.xlabel('Time')
    plt.ylabel('Entries')
    plt.show()

def plotPopTraject(newMatr):
    plt.scatter(newMatr[0,:], newMatr[1,:], c=np.arange(newMatr[0].shape[0]), cmap="viridis")
    plt.show()

X = np.loadtxt("Churchland_data.txt")
# In this dataset, n is the number of neurons and d is the number of time points
n, d = X.shape
# checkPCA(X)
U, S, V = PCA(X)
fracOfVar = frac_var_explained(S)
# plotFracVarExpl(fracOfVar)

UXbar = makeUXbar(X, U)

plotYvrsTime(UXbar)
plotPopTraject(UXbar)