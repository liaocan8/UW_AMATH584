import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt
import random

"""Part a"""
counter = 0
points = []
while counter < 1000:
    m = random.randint(1,1000)
    n = random.randint(1,m)
    A = npr.rand(m,n) # random m x n matrix with m > n
    K = npl.cond(A)
    point = (m,n,K)
    points.append(point)
    counter = counter + 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = [i[0] for i in points]
ys = [i[1] for i in points]
zs = [np.log(i[2]) for i in points]
ax.scatter(xs,ys,zs)
ax.set_xlabel("m")
ax.set_ylabel("n")
ax.set_zlabel("log(Condition Number)")
plt.show()

"""Part b"""
m = random.randint(1,100)
A = npr.rand(m,m)
KA = npl.cond(A)
detA = npl.det(A)
AT = A.T
AT[-1] = AT[0]
newA = AT.T
KnewA = npl.cond(newA)
detnewA = npl.det(newA)
print('------------------------------Part B------------------------------')
print(f"Condition number of A: {KA}")
print(f"Determinant of A: {detA}")
print(f"Condition number of A with last column replaced by first column: {KnewA}")
print(f"Determinant of A with last column replaced by first column: {detnewA}")
print("------------------------------------------------------------------")

"""Part c"""
e = [n*10**-14 for n in range(1,1000)]
noise_vector = npr.rand(1,m)
listKA = []
for n in e:
    M = AT
    M[-1] = M[0] + n*noise_vector
    listKA.append(npl.cond(M.T))

plt.scatter(e,listKA)
plt.show()