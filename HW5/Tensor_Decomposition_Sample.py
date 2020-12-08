import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import tensorly.decomposition as tld
import tensorly.tenalg as tlt

"""

We created two functions, f1 and f2, and created another function, F, by taking a linear combination of f1 and f2.
f1 is a Gaussian in the x and y direction that oscillates in time.
f2 is a "heart beat" in the x direction and a Gaussian in the y direction, both oscillating in time.

"""

def f1(x,y,t):
    return np.exp(-((x**2)+(y**2)))*np.cos(2*t)

def f2(x,y,t):
    return (1/np.cosh(x))*np.tanh(x)*np.exp(-0.2*y**2)*np.sin(t)

def F(x,y,t):
    return f1(x,y,t)+f2(x,y,t)

"""

X, Y, Z creates the 3D structure for F to work in. They are coordinate tensors.

"""

nx, ny, nt = (100, 100, 100)
x = np.linspace(-5, 5, nx)
y = np.linspace(-5,5,ny)
t = np.linspace(-10,10, nt)
X, Y, T = np.meshgrid(x,y,t)

"""

A is the tensor representation of F. To make A a tensor, F must be operated on the coordinate tensors created above to
inherit the tensor property.

tld.parafac does a "tensor SVD" on A. It returns two feature spaces embedded within F. In this case, the feature spaces
are f1 and f2.

tld.parafac returns (weights, factors)
    Weights is a 1D array of weights lambdaj where lambdaj is the weight for the tensor outer product aj o bj o cj that
    is in the linear decomposition of A.
    o indicates Kronecker product
    
    aj o bj o cj = fi(X,Y,T), for some i and j
    factors are matrices where factor[j] = [[aj],[bj],[cj]]. So factor[j] represents fi(X,Y,T)
    
    In the below case, factor2 represents f1 and factor1 represents f2.
    
"""

A = F(X,Y,T)
D = tld.parafac(A,2)
factor1 = np.array(D[1])[:,:,0]
factor2 = np.array(D[1])[:,:,1]

"""

Say factor j represents fi. Then The first row of factorj is the x projection of fi. Then the y projection and the
t projection.

When we plot the projects, they are consistent with projections of f1 and f2.

"""

plt.figure("x direction decomposition")
plt.plot(x,factor1[0,:])
plt.plot(x,factor2[0,:])

plt.figure("y direction decomposition")
plt.plot(y, factor1[1,:])
plt.plot(y, factor2[1,:])

plt.figure("t decomposition")
plt.plot(t,factor1[2,:])
plt.plot(t,factor2[2,:])
plt.show()