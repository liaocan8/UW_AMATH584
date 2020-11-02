import numpy as np
import numpy.linalg as npl

def QRfactor(A):
    """

    :param A: n x m Matrix as np array
    :return: [Q,R]
    QR = A
    Q is unitary
    R is upper triangular
    """

    size = A.shape
    columns_of_A = [A[:,i] for i in range(size[1])]
    index = 1
    Q_T = []

    while index <= size[1]:
        v1 = columns_of_A[index-1]
        v_counter = [v1]
        counter = 1
        while counter != index:
            v_counter.append(v_counter[counter-1] - np.dot(Q_T[counter-1],v_counter[counter-1])*Q_T[counter-1])
            counter = counter + 1
        Q_T.append(v_counter[-1] / npl.norm(v_counter[-1]))
        index = index + 1

    Q = np.array(Q_T).T
    R = np.matmul(Q.T,A)
    return (Q,R)

"""Testing on a well-conditioned matrix"""
B = np.random.rand(4,4)
K = npl.cond(B)
while K > 1000:
    B = np.random.rand(4,4)
    K = npl.cond(B)
QR = QRfactor(B)
Q = QR[0]
R = QR[1]
QTQ = np.matmul(Q.T,Q)

print("Testing on a well-conditioned matrix")
print("Below is Q")
print(Q)
print("-------------------------------------")
print("Below is R")
print(R)
print("-------------------------------------")
print("Below shows Q is unitary: Q.T Q = 1")
print(QTQ)
print("-------------------------------------")
print(f"Condition number of B is {K}")

"""Testing on ill-conditioned matrix"""
B2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
K2 = npl.cond(B2)
QR2 = QRfactor(B2)
Q2 = QR2[0]
R2 = QR2[1]
QTQ2 = np.matmul(Q2.T,Q2)
print("-------------------------------------")
print("-------------------------------------")
print("-------------------------------------")
print("Testing on a ill-conditioned matrix")
print("Below is Q")
print(Q2)
print("-------------------------------------")
print("Below is R")
print(R2)
print("-------------------------------------")
print("Below shows Q is unitary: Q.T Q = 1")
print(QTQ2)
print("-------------------------------------")
print(f"Condition number of B is {K2}")
