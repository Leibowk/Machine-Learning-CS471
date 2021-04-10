import numpy as np
import matplotlib.pyplot as plt

def true_fun(X):
    '''
    The true target function
    '''
    

    return np.cos(1.5 * np.pi * X[:,0]) - 0.3 * np.sin(2 * np.pi * X[:,1])

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

def showSVD(X):
    svd = np.linalg.svd(X)
    u, s, v = np.linalg.svd(X)

    xcords = []
    for i in range(len(s)):
        xcords.append(i)

    plt.plot(s)

    plt.xlabel('x cords')
    plt.ylabel('SVD')
    plt.yscale("log")

    plt.show()

def showSVDPoly(X,p):
    for i in range(p+1):
        print(i)
        featuresX = poly_features(X, i)
        u, s, v = np.linalg.svd(featuresX)
        print(s)
        print("\n \n")
        xcords = []
        for i in range(len(s)):
            xcords.append(i)
        
        plt.plot(xcords, s)

    plt.title("SVD of Degrees 0-10", fontsize=12)    
    plt.xlabel('x cords')
    plt.ylabel('SVD')
    plt.yscale("log")
    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    plt.show()


np.random.seed(0) # do not change this

n_samples = 40

X = np.random.rand(n_samples, 2)

showSVDPoly(X,10)

y = true_fun(X) + np.random.randn(n_samples) * 0.2

