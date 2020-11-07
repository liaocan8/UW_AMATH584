"""Decomposes a square matrix to a lower and an upper triangular matrix using pivoting"""

import numpy as np
import numpy.linalg as npl

def e(n,k):
    """

    :param n: dim of vector
    :param k: which dimension is the vector pointing in
    :return: np.array [[0], [0], [0], ... ,[0], [1], [0], ... , [0]]
    returns the unit vector for kth direction
    """
    v = np.zeros((1,n))
    v[0][k] = 1
    return np.array([v[0]]).T


def LUfactor(A):
    """

    :param A: n x n matrix
    :return: (L, U, P)
    PA = LU
    L is lower triangular
    U is upper triangular
    P is a permutation matrix
    """
    U = A
    n = A.shape[0]
    list_P = []
    list_Lk = []
    UT = U.T
    ColAsRow = UT.tolist()
    counter = 0
    while counter < n-1:
        """Computing U"""
        i = ColAsRow[counter]
        pivot = i[ColAsRow.index(i):][np.argmax(np.abs(i[ColAsRow.index(i):]))]

        current_pivot_index, new_pivot_index = ColAsRow.index(i), i.index(pivot)

        P = np.identity(n).tolist()
        P[current_pivot_index], P[new_pivot_index] = P[new_pivot_index], P[current_pivot_index]
        P = np.array(P)
        list_P.append(P) # storing Ps for later

        U = np.matmul(P,U) # must permute before finding L

        l = np.zeros((1,n)) # 2D np array row vector of zeros
        for j in range(current_pivot_index+1, n):
            l[0][j] = U[j][current_pivot_index] / pivot

        l = l.T

        L = np.identity(n) - np.matmul(l, e(n,current_pivot_index).T)
        list_Lk.append(L) # store L for later
        U = np.matmul(L,U)
        ColAsRow = U.T.tolist()
        counter += 1

    P = list_P[0]
    list_P.pop(0)
    for i in list_P:
        P = np.matmul(i, P)

    L = np.matmul(P, np.matmul(B, npl.inv(U)))

    return L, U, P


if __name__ == "__main__":
    B = np.random.rand(4,4)
    L,U,P= LUfactor(B)
    print('--------------------B---------------------------------')
    print(B)
    print('--------------------LU---------------------------------')
    print(L@U)
    print('--------------------PB---------------------------------')
    print(P@B)


