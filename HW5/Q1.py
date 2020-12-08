import numpy as np
import numpy.linalg as npl
import math as math

def rand_sym_matrix(n):
    """
    random symmetric int/half-int matrix generator
    :param n: size of matrix
    :return: symmetric matrix of size n
    """
    M = np.random.randint(-10, 10, (n,n))
    return (M+M.T)/2

def gershgorin_cicle(A):
    """

    :param A: m x m matrix
    :return: list of radii of Gershgorin Circles
    """
    R = []
    sizeA = A.shape[0]
    for i in range(sizeA):
        Ri = np.sum(abs(A[i,:]))-abs(A[i][i])
        R.append(Ri)
    return R

def power_iteration_sym(A):
    """

    :param A: square matrix
    :return: l, v
    l: largest eigenvalue; scaler
    v: eigenvector corresponding to l; 2darray column vector
    """
    sizeA = A.shape[0]
    v0 = np.zeros(sizeA)
    v0[0] = 1
    v0 = np.array([v0]).T
    w1 = np.matmul(A, v0)
    v1 = w1 / npl.norm(w1)
    v = [v0,v1]
    l0 = np.matmul(v[0].T,np.matmul(A,v[0])).item()
    l1 = np.matmul(v[1].T, np.matmul(A, v[1])).item()
    l = [l0,l1]
    counter = 2
    while abs(l[counter-1] - l[counter-2]) > 0.00000000001:
        next_w = np.matmul(A, v[counter-1])
        next_v = next_w / npl.norm(next_w)
        v.append(next_v)
        next_l = np.matmul(v[counter].T, np.matmul(A, v[counter])).item()
        l.append(next_l)
        counter += 1
    return l[-1], v[-1]

def power_iteration_general(A):
    sizeA = A.shape[0]
    v0 = np.random.rand(sizeA,1) + np.random.rand(sizeA,1) * 1j
    v0 = v0.astype(complex)
    w1 = np.matmul(A, v0)
    v1 = w1 / npl.norm(w1)
    v = [v0, v1]
    l0 = np.matmul(np.conj(v[0].T), np.matmul(A, v[0])).item()
    l1 = np.matmul(np.conj(v[1].T), np.matmul(A, v[1])).item()
    l = [l0, l1]
    counter = 2
    while abs(abs(l[counter - 1]) - abs(l[counter - 2])) > 0.000000001:
        if counter == 100:
            print("Oops")
            break
        next_w = np.matmul(A, v[counter - 1])
        next_v = next_w / npl.norm(next_w)
        v.append(next_v)
        next_l = np.matmul(np.conj(v[counter].T), np.matmul(A, v[counter])).item()
        l.append(next_l)
        counter += 1
    return l[-1], v[-1]


def rayleigh_quotient_iteration(A,search_range=30, number_steps=6666):
    """
    Uses Rayleigh quotient iteration to find the eigenvalue closest to mu and its corresponding eigenvector.
    :param A: m x m matrix
    :return: l, v
    l: eigenvalue closest to mu; scalar
    v: eigenvector corresponding to l; 2darray column vector
    """
    sizeA = A.shape[0]
    shifts = sorted(list(np.linspace(-search_range, search_range, num=number_steps)),key=abs)
    list_eigval = []
    list_eigvec = []
    newlist_eigval = []
    if all([A[i][j] == A[j][i] for i in range(sizeA) for j in range(sizeA)]):
        for k in shifts:
            mu = k
            v0 = np.zeros(sizeA)
            v0[0] = 1
            v0 = np.array([v0]).T
            pre_omega1 = npl.inv(A - mu*np.identity(sizeA))
            omega1 = np.matmul(pre_omega1,v0)
            v1 = omega1 / npl.norm(omega1)
            l1 = np.matmul(v0.T, np.matmul(A, v0)).item()
            v = [v0, v1]
            l = [mu,l1]
            counter = 2
            while abs(abs(l[counter-1]) - abs(l[counter-2])) > 0.00000001:
                pre_omega = npl.inv(A - l[counter-1]*np.identity(sizeA))
                omega = np.matmul(pre_omega,v[counter-1])
                next_v = omega / npl.norm(omega)
                v.append(next_v)
                next_l = np.matmul(v[counter].T, np.matmul(A, v[counter])).item()
                l.append(next_l)
                counter += 1
            eigval = l[-1]
            eigvec = v[-1]
            if any(abs(abs(x)-abs(eigval)) <= 0.00000001 for x in list_eigval):
                pass
            else:
                list_eigval.append(eigval)
                list_eigvec.append(eigvec)
            newlist_eigval = list_eigval
    else:
        for k in shifts:
            mu = k
            v0 = np.random.rand(sizeA, 1) + np.random.rand(sizeA, 1) * 1j
            v0 = v0.astype(complex)
            prepre_omega1 = A - mu * np.identity(sizeA)
            if math.isclose(npl.det(prepre_omega1).real,0) == True:
                break
            else:
                pre_omega1 = npl.inv(prepre_omega1)
            omega1 = np.matmul(pre_omega1, v0)
            v1 = omega1 / npl.norm(omega1)
            l1 = np.matmul(np.conj(v0.T), np.matmul(A, v0)).item()
            v = [v0, v1]
            l = [mu, l1]
            counter = 2
            while abs(abs(l[counter - 1]) - abs(l[counter - 2])) > 0.00000001:
                pre_omega = npl.inv(A - l[counter - 1] * np.identity(sizeA))
                omega = np.matmul(pre_omega, v[counter - 1])
                next_v = omega / npl.norm(omega)
                v.append(next_v)
                next_l = np.matmul(np.conj(v[counter].T), np.matmul(A, v[counter])).item()
                l.append(next_l)
                counter += 1
            eigval = l[-1]
            eigvec = v[-1]
            if any(abs(abs(x) - abs(eigval)) <= 0.00000001 for x in list_eigval):
                pass
            else:
                list_eigval.append(eigval)
                list_eigvec.append(eigvec)

        for p in list_eigval:

            newlist_eigval.append(p)
            if abs(p.imag) > 0.0001:
                newlist_eigval.append(np.conj(p))


    return newlist_eigval, list_eigvec


if __name__ == "__main__":
    A = rand_sym_matrix(10) # symmetric matrix
    B = np.array(np.random.randint(-10, 10, (10,10)),dtype=complex) # random matrix
    eigenvaluesA, eigenvectorsA = npl.eig(A)
    eigenvaluesB, eigenvectorsB = npl.eig(B)
    print("-------------------------------Eigenvalues of A from numpy.linalg.eig-------------------------------")
    print(eigenvaluesA)
    print("----------------------------Largest eigenvalue of A from power iteration----------------------------")
    print(power_iteration_sym(A)[0])
    print("------------------------All eigenvalues of A from Rayleigh Quotient Iteration-----------------------")
    print(rayleigh_quotient_iteration(A)[0])
    print("-------------------------------Eigenvalues of B from numpy.linalg.eig-------------------------------")
    print(eigenvaluesB)
    print("----------------------------Largest eigenvalue of B from power iteration----------------------------")
    print(power_iteration_general(B)[0])
    print("------------------------All eigenvalues of B from Rayleigh Quotient Iteration-----------------------")
    z = rayleigh_quotient_iteration(B, search_range=50, number_steps=6666)
    print(z[0])
    print("------------------------All eigenvectors of B from Rayleigh Quotient Iteration-----------------------")
    print(z[1])

    print("------------------------------------------")


