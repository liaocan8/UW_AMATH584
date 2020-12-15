import gzip
import numpy as np
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp

"""opens gz files"""
test_images = gzip.open('/home/liaocan8/Desktop/Library_of_Knowledge/UW classes/AMATH 584 Applied Linear Algebra/hw/hw6data/t10k-images-idx3-ubyte.gz', 'r')
test_labels = gzip.open('/home/liaocan8/Desktop/Library_of_Knowledge/UW classes/AMATH 584 Applied Linear Algebra/hw/hw6data/t10k-labels-idx1-ubyte.gz', 'r')
train_images = gzip.open('/home/liaocan8/Desktop/Library_of_Knowledge/UW classes/AMATH 584 Applied Linear Algebra/hw/hw6data/train-images-idx3-ubyte.gz', 'r')
train_labels = gzip.open('/home/liaocan8/Desktop/Library_of_Knowledge/UW classes/AMATH 584 Applied Linear Algebra/hw/hw6data/train-labels-idx1-ubyte.gz', 'r')

imsize = 28 # images were centred in a 28x28 frame
num_images_test = 10000 # num of images we want to read
num_images_train = 60000

list_test_images = []
test_images.read(16) # skips non-image info
buf_test_images = test_images.read(imsize*imsize*num_images_test) # represents the data in bytes
data_test_images = np.frombuffer(buf_test_images, dtype=np.uint8).astype(np.float32) # converts data in bytes to nparray
data_test_images = data_test_images.reshape(num_images_test, imsize, imsize, 1) # data is stored as a hypercube
for i in data_test_images:
    image = np.asarray(i).squeeze()
    list_test_images.append(image)

list_train_images = []
train_images.read(16)
buf_train_images = train_images.read(imsize*imsize*num_images_train)
data_train_images = np.frombuffer(buf_train_images, dtype=np.uint8).astype(np.float32)
data_train_images = data_train_images.reshape(num_images_train, imsize, imsize, 1)
for i in data_train_images:
    image = np.asarray(i).squeeze()
    list_train_images.append(image)

test_labels.read(8) # skips useless info
buf_test_labels = test_labels.read() # opens data in bytes
list_test_labels = np.frombuffer(buf_test_labels, dtype=np.uint8).astype(np.int64) # turns bytes to 1Darray

train_labels.read(8)
buf_train_labels = train_labels.read()
list_train_labels = np.frombuffer(buf_train_labels, dtype=np.uint8).astype(np.int64)

B_test = [] # label matrix
for i in list_test_labels:
    v = list(np.zeros(10))
    v[i-1] = 1
    B_test.append(v)
B_test = np.array(B_test).T

B_train = [] # label matrix
for i in list_train_labels:
    v = list(np.zeros(10))
    v[i-1] = 1
    B_train.append(v)
B_train = np.array(B_train).T

vectorized_test_images = [i.flatten() for i in list_test_images]
vectorized_train_images = [i.flatten() for i in list_train_images]

average_train_image = np.sum(list_train_images,0) / len(list_train_images) # getting to average image to centre data
vectorized_average_train_image = average_train_image.flatten()

A_test = np.array([i - vectorized_average_train_image for i in vectorized_test_images]).T
A_train = np.array([i - vectorized_average_train_image for i in vectorized_train_images]).T

def to_label(X):
    """
    Converts 10 x n label matrix into a 1darray of labels
    :param X: 10 x n label matrix
    :return: 1darray of labels
    """
    XT = list(X.T)
    XTcompletelist = [list(i) for i in XT]
    list_of_label_vectors = []
    for i in range(10):
        vector = list(np.zeros(10))
        vector[i] = 1
        list_of_label_vectors.append(vector)
    labellist = []
    for i in XTcompletelist:
        if np.linalg.norm(i) == 1:
            for j, vector in enumerate(list_of_label_vectors):
                if i == vector:
                    if j+1 == 10:
                        labellist.append(0)
                    elif j+1 != 10:
                        labellist.append(j+1)
                else:
                    continue
        else:
            labellist.append(11) # 11 means the vector did not match a label vector

    return labellist

def rounding_label_matrix(B):
    BT = B.T
    listB = list(BT)
    for i in listB:
        for j in i:
            loc = np.where(i == j)[0].item()
            if abs(j) == np.amax(i):
                i[loc] = 1
            else:
                i[loc] = 0
    newB = np.array(listB).T
    return newB


"""To get map A to B, we must solve for X in XA=B. To use numpy.linalg.lstsq to solve for X, we solve for B.T = A.T * X.T"""
XT_numpylinalgsolver = np.linalg.lstsq(A_train.T, B_train.T, rcond=None) # solving using numpy.linalg.lstsq
X_nplinalglstsq = XT_numpylinalgsolver[0].T
fig_nplinalglstsq = plt.figure("numpy.linalg.lstsq")
im_nplinalglstsq_1 = fig_nplinalglstsq.add_subplot(5,5,2)
im_nplinalglstsq_2 = fig_nplinalglstsq.add_subplot(5,5,4)
im_nplinalglstsq_3 = fig_nplinalglstsq.add_subplot(5,5,6)
im_nplinalglstsq_4 = fig_nplinalglstsq.add_subplot(5,5,8)
im_nplinalglstsq_5 = fig_nplinalglstsq.add_subplot(5,5,10)
im_nplinalglstsq_6 = fig_nplinalglstsq.add_subplot(5,5,12)
im_nplinalglstsq_7 = fig_nplinalglstsq.add_subplot(5,5,14)
im_nplinalglstsq_8 = fig_nplinalglstsq.add_subplot(5,5,16)
im_nplinalglstsq_9 = fig_nplinalglstsq.add_subplot(5,5,18)
im_nplinalglstsq_10 = fig_nplinalglstsq.add_subplot(5,5,20)
im_nplinalglstsq_1.imshow(np.reshape(X_nplinalglstsq[0,:],(imsize,imsize)))
im_nplinalglstsq_1.set_title("1")
im_nplinalglstsq_2.imshow(np.reshape(X_nplinalglstsq[1,:],(imsize,imsize)))
im_nplinalglstsq_2.set_title("2")
im_nplinalglstsq_3.imshow(np.reshape(X_nplinalglstsq[2,:],(imsize,imsize)))
im_nplinalglstsq_3.set_title("3")
im_nplinalglstsq_4.imshow(np.reshape(X_nplinalglstsq[3,:],(imsize,imsize)))
im_nplinalglstsq_4.set_title("4")
im_nplinalglstsq_5.imshow(np.reshape(X_nplinalglstsq[4,:],(imsize,imsize)))
im_nplinalglstsq_5.set_title("5")
im_nplinalglstsq_6.imshow(np.reshape(X_nplinalglstsq[5,:],(imsize,imsize)))
im_nplinalglstsq_6.set_title("6")
im_nplinalglstsq_7.imshow(np.reshape(X_nplinalglstsq[6,:],(imsize,imsize)))
im_nplinalglstsq_7.set_title("7")
im_nplinalglstsq_8.imshow(np.reshape(X_nplinalglstsq[7,:],(imsize,imsize)))
im_nplinalglstsq_8.set_title("8")
im_nplinalglstsq_9.imshow(np.reshape(X_nplinalglstsq[8,:],(imsize,imsize)))
im_nplinalglstsq_9.set_title("9")
im_nplinalglstsq_10.imshow(np.reshape(X_nplinalglstsq[9,:],(imsize,imsize)))
im_nplinalglstsq_10.set_title("0")

model_sklearnlasso_alpha1 = skl.Lasso(alpha=1) # solving using sklearn.linear_model_Lasso with sparsity parameter at 1
model_sklearnlasso_alpha1.fit(A_train.T, B_train.T)
X_sklearnlasso_alpha1 = model_sklearnlasso_alpha1.coef_
fig_alpha1 = plt.figure("Lasso alpha=1 : Most Important Pixels for Each Digit")
im_alpha1_1 = fig_alpha1.add_subplot(5,5,1)
im_alpha1_2 = fig_alpha1.add_subplot(5,5,3)
im_alpha1_3 = fig_alpha1.add_subplot(5,5,5)
im_alpha1_4 = fig_alpha1.add_subplot(5,5,7)
im_alpha1_5 = fig_alpha1.add_subplot(5,5,9)
im_alpha1_6 = fig_alpha1.add_subplot(5,5,11)
im_alpha1_7 = fig_alpha1.add_subplot(5,5,13)
im_alpha1_8 = fig_alpha1.add_subplot(5,5,15)
im_alpha1_9 = fig_alpha1.add_subplot(5,5,17)
im_alpha1_10 = fig_alpha1.add_subplot(5,5,19)
im_alpha1_1.imshow(np.reshape(X_sklearnlasso_alpha1[0,:],(imsize,imsize)))
im_alpha1_1.set_title("1")
im_alpha1_2.imshow(np.reshape(X_sklearnlasso_alpha1[1,:],(imsize,imsize)))
im_alpha1_2.set_title("2")
im_alpha1_3.imshow(np.reshape(X_sklearnlasso_alpha1[2,:],(imsize,imsize)))
im_alpha1_3.set_title("3")
im_alpha1_4.imshow(np.reshape(X_sklearnlasso_alpha1[3,:],(imsize,imsize)))
im_alpha1_4.set_title("4")
im_alpha1_5.imshow(np.reshape(X_sklearnlasso_alpha1[4,:],(imsize,imsize)))
im_alpha1_5.set_title("5")
im_alpha1_6.imshow(np.reshape(X_sklearnlasso_alpha1[5,:],(imsize,imsize)))
im_alpha1_6.set_title("6")
im_alpha1_7.imshow(np.reshape(X_sklearnlasso_alpha1[6,:],(imsize,imsize)))
im_alpha1_7.set_title("7")
im_alpha1_8.imshow(np.reshape(X_sklearnlasso_alpha1[7,:],(imsize,imsize)))
im_alpha1_8.set_title("8")
im_alpha1_9.imshow(np.reshape(X_sklearnlasso_alpha1[8,:],(imsize,imsize)))
im_alpha1_9.set_title("9")
im_alpha1_10.imshow(np.reshape(X_sklearnlasso_alpha1[9,:],(imsize,imsize)))
im_alpha1_10.set_title("0")

model_sklearnlasso_alpha05 = skl.Lasso(alpha=0.5) # solving using sklearn.linear_model_Lasso with sparsity parameter at 0.5
model_sklearnlasso_alpha05.fit(A_train.T, B_train.T)
X_sklearnlasso_alpha05 = model_sklearnlasso_alpha05.coef_
fig_alpha05 = plt.figure("Lasso alpha=0.5")
im_alpha05_1 = fig_alpha05.add_subplot(5,5,1)
im_alpha05_2 = fig_alpha05.add_subplot(5,5,3)
im_alpha05_3 = fig_alpha05.add_subplot(5,5,5)
im_alpha05_4 = fig_alpha05.add_subplot(5,5,7)
im_alpha05_5 = fig_alpha05.add_subplot(5,5,9)
im_alpha05_6 = fig_alpha05.add_subplot(5,5,11)
im_alpha05_7 = fig_alpha05.add_subplot(5,5,13)
im_alpha05_8 = fig_alpha05.add_subplot(5,5,15)
im_alpha05_9 = fig_alpha05.add_subplot(5,5,17)
im_alpha05_10 = fig_alpha05.add_subplot(5,5,19)
im_alpha05_1.imshow(np.reshape(X_sklearnlasso_alpha05[0,:],(imsize,imsize)))
im_alpha05_1.set_title("1")
im_alpha05_2.imshow(np.reshape(X_sklearnlasso_alpha05[1,:],(imsize,imsize)))
im_alpha05_2.set_title("2")
im_alpha05_3.imshow(np.reshape(X_sklearnlasso_alpha05[2,:],(imsize,imsize)))
im_alpha05_3.set_title("3")
im_alpha05_4.imshow(np.reshape(X_sklearnlasso_alpha05[3,:],(imsize,imsize)))
im_alpha05_4.set_title("4")
im_alpha05_5.imshow(np.reshape(X_sklearnlasso_alpha05[4,:],(imsize,imsize)))
im_alpha05_5.set_title("5")
im_alpha05_6.imshow(np.reshape(X_sklearnlasso_alpha05[5,:],(imsize,imsize)))
im_alpha05_6.set_title("6")
im_alpha05_7.imshow(np.reshape(X_sklearnlasso_alpha05[6,:],(imsize,imsize)))
im_alpha05_7.set_title("7")
im_alpha05_8.imshow(np.reshape(X_sklearnlasso_alpha05[7,:],(imsize,imsize)))
im_alpha05_8.set_title("8")
im_alpha05_9.imshow(np.reshape(X_sklearnlasso_alpha05[8,:],(imsize,imsize)))
im_alpha05_9.set_title("9")
im_alpha05_10.imshow(np.reshape(X_sklearnlasso_alpha05[9,:],(imsize,imsize)))
im_alpha05_10.set_title("0")

model_sklearnlasso_alpha01 = skl.Lasso(alpha=0.1) # solving using sklearn.linear_model_Lasso with sparsity parameter at 0.1
model_sklearnlasso_alpha01.fit(A_train.T, B_train.T)
X_sklearnlasso_alpha01 = model_sklearnlasso_alpha01.coef_
fig_alpha01 = plt.figure("Lasso alpha=0.1")
im_alpha01_1 = fig_alpha01.add_subplot(5,5,1)
im_alpha01_2 = fig_alpha01.add_subplot(5,5,3)
im_alpha01_3 = fig_alpha01.add_subplot(5,5,5)
im_alpha01_4 = fig_alpha01.add_subplot(5,5,7)
im_alpha01_5 = fig_alpha01.add_subplot(5,5,9)
im_alpha01_6 = fig_alpha01.add_subplot(5,5,11)
im_alpha01_7 = fig_alpha01.add_subplot(5,5,13)
im_alpha01_8 = fig_alpha01.add_subplot(5,5,15)
im_alpha01_9 = fig_alpha01.add_subplot(5,5,17)
im_alpha01_10 = fig_alpha01.add_subplot(5,5,19)
im_alpha01_1.imshow(np.reshape(X_sklearnlasso_alpha01[0,:],(imsize,imsize)))
im_alpha01_1.set_title("1")
im_alpha01_2.imshow(np.reshape(X_sklearnlasso_alpha01[1,:],(imsize,imsize)))
im_alpha01_2.set_title("2")
im_alpha01_3.imshow(np.reshape(X_sklearnlasso_alpha01[2,:],(imsize,imsize)))
im_alpha01_3.set_title("3")
im_alpha01_4.imshow(np.reshape(X_sklearnlasso_alpha01[3,:],(imsize,imsize)))
im_alpha01_4.set_title("4")
im_alpha01_5.imshow(np.reshape(X_sklearnlasso_alpha01[4,:],(imsize,imsize)))
im_alpha01_5.set_title("5")
im_alpha01_6.imshow(np.reshape(X_sklearnlasso_alpha01[5,:],(imsize,imsize)))
im_alpha01_6.set_title("6")
im_alpha01_7.imshow(np.reshape(X_sklearnlasso_alpha01[6,:],(imsize,imsize)))
im_alpha01_7.set_title("7")
im_alpha01_8.imshow(np.reshape(X_sklearnlasso_alpha01[7,:],(imsize,imsize)))
im_alpha01_8.set_title("8")
im_alpha01_9.imshow(np.reshape(X_sklearnlasso_alpha01[8,:],(imsize,imsize)))
im_alpha01_9.set_title("9")
im_alpha01_10.imshow(np.reshape(X_sklearnlasso_alpha01[9,:],(imsize,imsize)))
im_alpha01_10.set_title("0")

model_sklearnlasso_alpha001 = skl.Lasso(alpha=0.01) # solving using sklearn.linear_model_Lasso with sparsity parameter at 0.01
model_sklearnlasso_alpha001.fit(A_train.T, B_train.T)
X_sklearnlasso_alpha001 = model_sklearnlasso_alpha001.coef_
fig_alpha001 = plt.figure("Lasso alpha=0.01")
im_alpha001_1 = fig_alpha001.add_subplot(5,5,1)
im_alpha001_2 = fig_alpha001.add_subplot(5,5,3)
im_alpha001_3 = fig_alpha001.add_subplot(5,5,5)
im_alpha001_4 = fig_alpha001.add_subplot(5,5,7)
im_alpha001_5 = fig_alpha001.add_subplot(5,5,9)
im_alpha001_6 = fig_alpha001.add_subplot(5,5,11)
im_alpha001_7 = fig_alpha001.add_subplot(5,5,13)
im_alpha001_8 = fig_alpha001.add_subplot(5,5,15)
im_alpha001_9 = fig_alpha001.add_subplot(5,5,17)
im_alpha001_10 = fig_alpha001.add_subplot(5,5,19)
im_alpha001_1.imshow(np.reshape(X_sklearnlasso_alpha001[0,:],(imsize,imsize)))
im_alpha001_1.set_title("1")
im_alpha001_2.imshow(np.reshape(X_sklearnlasso_alpha001[1,:],(imsize,imsize)))
im_alpha001_2.set_title("2")
im_alpha001_3.imshow(np.reshape(X_sklearnlasso_alpha001[2,:],(imsize,imsize)))
im_alpha001_3.set_title("3")
im_alpha001_4.imshow(np.reshape(X_sklearnlasso_alpha001[3,:],(imsize,imsize)))
im_alpha001_4.set_title("4")
im_alpha001_5.imshow(np.reshape(X_sklearnlasso_alpha001[4,:],(imsize,imsize)))
im_alpha001_5.set_title("5")
im_alpha001_6.imshow(np.reshape(X_sklearnlasso_alpha001[5,:],(imsize,imsize)))
im_alpha001_6.set_title("6")
im_alpha001_7.imshow(np.reshape(X_sklearnlasso_alpha001[6,:],(imsize,imsize)))
im_alpha001_7.set_title("7")
im_alpha001_8.imshow(np.reshape(X_sklearnlasso_alpha001[7,:],(imsize,imsize)))
im_alpha001_8.set_title("8")
im_alpha001_9.imshow(np.reshape(X_sklearnlasso_alpha001[8,:],(imsize,imsize)))
im_alpha001_9.set_title("9")
im_alpha001_10.imshow(np.reshape(X_sklearnlasso_alpha001[9,:],(imsize,imsize)))
im_alpha001_10.set_title("0")

model_sklearnridge_alpha1 = skl.Ridge(alpha=1) # solving using sklearn.linear_model_Ridge with alpha=1
model_sklearnridge_alpha1.fit(A_train.T, B_train.T)
X_sklearnridge_alpha1 = model_sklearnridge_alpha1.coef_
fig_ridge_alpha1 = plt.figure("Ridge alpha=1")
im_ridge_alpha1_1 = fig_ridge_alpha1.add_subplot(5,5,1)
im_ridge_alpha1_2 = fig_ridge_alpha1.add_subplot(5,5,3)
im_ridge_alpha1_3 = fig_ridge_alpha1.add_subplot(5,5,5)
im_ridge_alpha1_4 = fig_ridge_alpha1.add_subplot(5,5,7)
im_ridge_alpha1_5 = fig_ridge_alpha1.add_subplot(5,5,9)
im_ridge_alpha1_6 = fig_ridge_alpha1.add_subplot(5,5,11)
im_ridge_alpha1_7 = fig_ridge_alpha1.add_subplot(5,5,13)
im_ridge_alpha1_8 = fig_ridge_alpha1.add_subplot(5,5,15)
im_ridge_alpha1_9 = fig_ridge_alpha1.add_subplot(5,5,17)
im_ridge_alpha1_10 = fig_ridge_alpha1.add_subplot(5,5,19)
im_ridge_alpha1_1.imshow(np.reshape(X_sklearnridge_alpha1[0,:],(imsize,imsize)))
im_ridge_alpha1_1.set_title("1")
im_ridge_alpha1_2.imshow(np.reshape(X_sklearnridge_alpha1[1,:],(imsize,imsize)))
im_ridge_alpha1_2.set_title("2")
im_ridge_alpha1_3.imshow(np.reshape(X_sklearnridge_alpha1[2,:],(imsize,imsize)))
im_ridge_alpha1_3.set_title("3")
im_ridge_alpha1_4.imshow(np.reshape(X_sklearnridge_alpha1[3,:],(imsize,imsize)))
im_ridge_alpha1_4.set_title("4")
im_ridge_alpha1_5.imshow(np.reshape(X_sklearnridge_alpha1[4,:],(imsize,imsize)))
im_ridge_alpha1_5.set_title("5")
im_ridge_alpha1_6.imshow(np.reshape(X_sklearnridge_alpha1[5,:],(imsize,imsize)))
im_ridge_alpha1_6.set_title("6")
im_ridge_alpha1_7.imshow(np.reshape(X_sklearnridge_alpha1[6,:],(imsize,imsize)))
im_ridge_alpha1_7.set_title("7")
im_ridge_alpha1_8.imshow(np.reshape(X_sklearnridge_alpha1[7,:],(imsize,imsize)))
im_ridge_alpha1_8.set_title("8")
im_ridge_alpha1_9.imshow(np.reshape(X_sklearnridge_alpha1[8,:],(imsize,imsize)))
im_ridge_alpha1_9.set_title("9")
im_ridge_alpha1_10.imshow(np.reshape(X_sklearnridge_alpha1[9,:],(imsize,imsize)))
im_ridge_alpha1_10.set_title("0")

model_sklearnLR = skl.LinearRegression() # solving using sklearn.linear_model_LinearRegression
model_sklearnLR.fit(A_train.T, B_train.T)
X_sklearnLR = model_sklearnLR.coef_
fig_LR = plt.figure("LinearRegression")
im_LR_1 = fig_LR.add_subplot(5,5,1)
im_LR_2 = fig_LR.add_subplot(5,5,3)
im_LR_3 = fig_LR.add_subplot(5,5,5)
im_LR_4 = fig_LR.add_subplot(5,5,7)
im_LR_5 = fig_LR.add_subplot(5,5,9)
im_LR_6 = fig_LR.add_subplot(5,5,11)
im_LR_7 = fig_LR.add_subplot(5,5,13)
im_LR_8 = fig_LR.add_subplot(5,5,15)
im_LR_9 = fig_LR.add_subplot(5,5,17)
im_LR_10 = fig_LR.add_subplot(5,5,19)
im_LR_1.imshow(np.reshape(X_sklearnLR[0,:],(imsize,imsize)))
im_LR_1.set_title("1")
im_LR_2.imshow(np.reshape(X_sklearnLR[1,:],(imsize,imsize)))
im_LR_2.set_title("2")
im_LR_3.imshow(np.reshape(X_sklearnLR[2,:],(imsize,imsize)))
im_LR_3.set_title("3")
im_LR_4.imshow(np.reshape(X_sklearnLR[3,:],(imsize,imsize)))
im_LR_4.set_title("4")
im_LR_5.imshow(np.reshape(X_sklearnLR[4,:],(imsize,imsize)))
im_LR_5.set_title("5")
im_LR_6.imshow(np.reshape(X_sklearnLR[5,:],(imsize,imsize)))
im_LR_6.set_title("6")
im_LR_7.imshow(np.reshape(X_sklearnLR[6,:],(imsize,imsize)))
im_LR_7.set_title("7")
im_LR_8.imshow(np.reshape(X_sklearnLR[7,:],(imsize,imsize)))
im_LR_8.set_title("8")
im_LR_9.imshow(np.reshape(X_sklearnLR[8,:],(imsize,imsize)))
im_LR_9.set_title("9")
im_LR_10.imshow(np.reshape(X_sklearnLR[9,:],(imsize,imsize)))
im_LR_10.set_title("0")

"""Applying model to test images."""
testB_nplinalglstsq = rounding_label_matrix(np.matmul(X_nplinalglstsq, A_test))
testlabels_nplinalglstsq = to_label(testB_nplinalglstsq)
testB_sklearnlasso_alpha1 = rounding_label_matrix(np.matmul(X_sklearnlasso_alpha1, A_test))
testlabels_sklearnlasso_alpha1 = to_label(testB_sklearnlasso_alpha1)
testB_sklearnridge_alpha1 = rounding_label_matrix(np.matmul(X_sklearnridge_alpha1, A_test))
testlabels_sklearnridge_alpha1 = to_label(testB_sklearnridge_alpha1)
testB_sklearnLR = rounding_label_matrix(np.matmul(X_sklearnLR, A_test))
testlabels_sklearnLR = to_label(testB_sklearnLR)

"""
Comparing experimental labels to actual labels.
If entry i is correct, testcompare[i] = True.
If entry i is incorrect, testcompare[i] = False.
"""
testcompare_nplinalglstsq = np.array(testlabels_nplinalglstsq) == list_test_labels
testcompare_sklearnlasso_alpha1 = np.array(testlabels_sklearnlasso_alpha1) == list_test_labels
testcompare_sklearnridge_alpha1 = np.array(testlabels_sklearnridge_alpha1) == list_test_labels
testcompare_sklearnLR = np.array(testlabels_sklearnLR) == list_test_labels

"""Counting the number of True or False"""
unique_nplinalglstsq, counts_nplinalglstsq = np.unique(testcompare_nplinalglstsq, return_counts=True)
unique_sklearnlasso_alpha1, counts_sklearnlasso_alpha1 = np.unique(testcompare_sklearnlasso_alpha1, return_counts=True)
unique_sklearnridge_alpha1, counts_sklearnridge_alpha1 = np.unique(testcompare_sklearnridge_alpha1, return_counts=True)
unique_sklearnLR, counts_sklearnLR = np.unique(testcompare_sklearnLR, return_counts=True)

"""Converting results into a dict."""
results_nplinalglstsq = dict(zip(unique_nplinalglstsq, counts_nplinalglstsq))
results_sklearnlasso_alpha1 = dict(zip(unique_sklearnlasso_alpha1, counts_sklearnlasso_alpha1))
results_sklearnridge_alpha1 = dict(zip(unique_sklearnridge_alpha1, counts_sklearnridge_alpha1))
results_sklearnLR = dict(zip(unique_sklearnLR, counts_sklearnLR))

"""
Applying the same analysis to each digit individually to find the most important pixels. 
scipy.sparse.linalg.lsqr had to be used since it won't converge for sklearn.linear_model.Lasso
"""

def extract_digits(n,testortrain): # returns list of images of digit n in test/train data
    extracted = []
    if testortrain == 'train':
        index = np.where(list_train_labels==n)[0]
        for i in index:
            extracted.append(list_train_images[i])
    elif testortrain == 'test':
        index = np.where(list_test_labels==n)[0]
        for i in index:
            extracted.append(list_test_images[i])
    return extracted

def normalized_A(n, testortrain):
    listn = extract_digits(n, testortrain)
    average_nimage = np.sum(listn,0) / len(listn)
    preAn = [i.flatten() - average_nimage.flatten() for i in listn]
    An = np.array(preAn).T
    return An

trainAn = [normalized_A(n,'train') for n in range(10)]
trainBn = [np.ones(i.shape[1]) for i in trainAn]
Xn = [sp.lsqr(trainAn[i].T,trainBn[i])[0] for i in range(10)] # are (784,) 1d arrays
fig_indivdigit = plt.figure("Most Important Pixels Obtained Using Individual Digits")
im_indivdigit_1 = fig_indivdigit.add_subplot(5,5,2)
im_indivdigit_2 = fig_indivdigit.add_subplot(5,5,4)
im_indivdigit_3 = fig_indivdigit.add_subplot(5,5,6)
im_indivdigit_4 = fig_indivdigit.add_subplot(5,5,8)
im_indivdigit_5 = fig_indivdigit.add_subplot(5,5,10)
im_indivdigit_6 = fig_indivdigit.add_subplot(5,5,12)
im_indivdigit_7 = fig_indivdigit.add_subplot(5,5,14)
im_indivdigit_8 = fig_indivdigit.add_subplot(5,5,16)
im_indivdigit_9 = fig_indivdigit.add_subplot(5,5,18)
im_indivdigit_10 = fig_indivdigit.add_subplot(5,5,20)
im_indivdigit_1.imshow(np.reshape(Xn[1],(imsize,imsize)))
im_indivdigit_1.set_title("1")
im_indivdigit_2.imshow(np.reshape(Xn[2],(imsize,imsize)))
im_indivdigit_2.set_title("2")
im_indivdigit_3.imshow(np.reshape(Xn[3],(imsize,imsize)))
im_indivdigit_3.set_title("3")
im_indivdigit_4.imshow(np.reshape(Xn[4],(imsize,imsize)))
im_indivdigit_4.set_title("4")
im_indivdigit_5.imshow(np.reshape(Xn[5],(imsize,imsize)))
im_indivdigit_5.set_title("5")
im_indivdigit_6.imshow(np.reshape(Xn[6],(imsize,imsize)))
im_indivdigit_6.set_title("6")
im_indivdigit_7.imshow(np.reshape(Xn[7],(imsize,imsize)))
im_indivdigit_7.set_title("7")
im_indivdigit_8.imshow(np.reshape(Xn[8],(imsize,imsize)))
im_indivdigit_8.set_title("8")
im_indivdigit_9.imshow(np.reshape(Xn[9],(imsize,imsize)))
im_indivdigit_9.set_title("9")
im_indivdigit_10.imshow(np.reshape(Xn[0],(imsize,imsize)))
im_indivdigit_10.set_title("0")


"""
Solvers that were used:
numpy.linalg.lstsq
sklearn.linear_model.Lasso with alpha = 1, 0.75, 0.5, 0.1, 0.01
sklearn.linear_model.Ridge with alpha = 1
sklearn.linear_model.LinearRegression

The ridge regression can't be trusted because our matrix is ill-conditioned.

By looking at the images resulting from solving XA=B with various L1 penalty coefficients, we can see which pixels are 
important in determining the number written. The more important the pixels stay nonzero as alpha increases.

The only most important pixels (from sklearn.linear_model.Lasso(alpha=0.01)) were used in identifying the test images.
8332/10000 were correctly identified, under-performing the results using other solvers. 

When the most important pixels of the digits were obtained individually, more pixels were included.

In this problem, we are using XA=B to create a linear map between our matrix of images, A, and the matrix representing
the label for these images, B.

"""

print(results_nplinalglstsq, "numpy.linalg.lstsq results")
print(results_sklearnlasso_alpha1, "sklearn.linear_model.Lasso alpha=1 results")
print(results_sklearnridge_alpha1, "sklearn.linear_model.Ridge alpha=1 results")
print(results_sklearnLR, "sklearn.linear_model.LinearRegression results")
print("More to say at the bottom of the code.")

plt.show()

test_images.close()
test_labels.close()
train_images.close()
train_labels.close()
