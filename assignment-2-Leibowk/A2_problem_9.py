import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import timeit
import time

def solve_system_inv(n):
    '''
    Solve a random linear system Ax = b, where A and b are random.

    Parameters: 
    n (int): size of linear system n x n

    Returns:
    A, b, x (np.ndarray): system matrix, rhs, solution
    '''

    A = np.random.randn(n, n)
    b = np.random.randn(n)

    # Your code here to solve A x = b

    # Form the inverse of A and use it to compute the solution (9.1)
    Ainv= np.linalg.inv(A)
    x= np.multiply(Ainv,b)

def solve_system_pinv(n):
    '''
    Solve a random linear system Ax = b, where A and b are random.

    Parameters: 
    n (int): size of linear system n x n

    Returns:
    A, b, x (np.ndarray): system matrix, rhs, solution
    '''

    A = np.random.randn(n, n)
    b = np.random.randn(n)

    # Your code here to solve A x = b
    # Form the pseudoinverse AT and compute the pseudoinverse solution. (9.2)
    pinv = np.linalg.pinv(A)
    Ainv= np.linalg.inv(A)
    x = np.multiply(Ainv,pinv)
    return (A, b, x)

def solve_system_numpy(n):
    '''
    Solve a random linear system Ax = b, where A and b are random.

    Parameters: 
    n (int): size of linear system n x n

    Returns:
    A, b, x (np.ndarray): system matrix, rhs, solution
    '''

    A = np.random.randn(n, n)
    b = np.random.randn(n)

    # Using the optimized function np.linalg.solve (9.3)
    x= np.linalg.solve(A,b)

    return (A, b, x)

# def timeIt(func):
#     start_time = time.time()
#     func
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     return elapsed_time

xcords = []
y1 = []
y2 = []
y3 = []

plt.xlabel('n')
plt.ylabel('time ellapsed')
plt.yscale("log")
plt.xscale("log")

# print(timeIt(solve_system_inv(1000)))

for x in range(10,1000):
    xcords.append(x)
    
    a = timeit.timeit('solve_system_inv(x)','from __main__ import solve_system_inv, x', number=10)
    # a = 0
    # for x in range(10):
    #     a = a + timeit.timeit('solve_system_inv(x)','from __main__ import solve_system_inv, x', number=1)
    a = a/10
    y1.append(a)
    
    b = timeit.timeit('solve_system_pinv(x)','from __main__ import solve_system_pinv, x', number=10)
    # b = 0
    # for x in range(10):
        # b = b + timeIt(solve_system_pinv(x))
        # b = b + timeit.timeit('solve_system_pinv(x)','from __main__ import solve_system_pinv, x', number=1)
    b = b/10 
    y2.append(b)
    
    c = timeit.timeit('solve_system_numpy(x)','from __main__ import solve_system_numpy, x', number=10)
    # c = 0
    # for x in range(10):
    #     # c = c + timeIt(solve_system_numpy(x))
    #      c = c + timeit.timeit('solve_system_numpy(x)','from __main__ import solve_system_numpy, x', number=1)
    c = c/10 
    y3.append(c)

plt.plot(xcords, y1)
plt.plot(xcords, y2)
plt.plot(xcords, y3)

plt.legend(["Inv", "Pinv", "Numpy"])

plt.show()
    
