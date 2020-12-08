from skimage.io import imread
import numpy as np
import numpy.linalg as npl
import matplotlib.pylab as plt
import os as oss
import os.path as op

def dir_list(folder):
    """

    :param folder: A folder containing files
    :return: A list of strings where the strings are Python readable file names of the files in folder
    """
    top = r'/home/liaocan8/Desktop/Library_of_Knowledge/UW classes/AMATH 584 Applied Linear Algebra/hw/hw 2 data/cropped/CroppedYale'
    dir = op.join(top, folder)
    walker = list(oss.walk(dir))
    list_of_images = []
    for i in walker[0][2]:
        list_of_images.append(op.join(walker[0][0],i))
    return list_of_images

def image_list(folder):
    """

    :param folder: A folder containing images
    :return: List of images already imported using imread
    """
    list_of_images = []
    for i in dir_list(folder):
        list_of_images.append(imread(i))
    return list_of_images

def average_face(imagelist):
    """

    :param imagelist: list of images
    :return: averaged image of images in the folder
    """
    a_face = sum(imagelist)/len(imagelist)
    return a_face

def power_iteration(A):
    sizeA = A.shape[1]
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

def random_normalized_matrix(n, m):
    """

    :param n: dimension of vectors (rows of matrix)
    :param m: how many vectors (columns of matrix)
    :return: n x m matrix with its columns as normalized vectors
    """
    D = []
    for i in range(m):
        v = np.random.rand(n)
        v = v / npl.norm(v)
        D.append(v)
    D = np.array(D).T
    return D

def randomized_SVD(A, r):
    """

    :param A: n x m data matrix where each column is an entry
    :param r: number of randomized samples
    :return: U, S, Vh
    """
    n, m = A.shape
    omega = random_normalized_matrix(m, r)
    Y = np.matmul(A, omega)

    QR = npl.qr(Y)
    Q = QR[0]
    B = np.matmul(Q.T, A)

    Utilde, S, Vh = npl.svd(B, full_matrices=False)
    U = np.matmul(Q, Utilde)
    return U, S, Vh

if __name__ == '__main__':
    images_by_folder = [image_list(i) for i in [f'yaleB{n}' for n in range(1, 39)]]
    # makes a list of lists of images where the sub-list is a list of images in yaleB{n}
    all_images = []  # all training images
    for i in images_by_folder:
        for j in i:
            all_images.append(j)
    im_shape = all_images[0].shape  # shape of an image

    A = np.array([i.flatten() for i in all_images]).T  # data matrix where each column is an image
    A_face_vector = np.array([np.sum(A, axis=1) / len(all_images)]).T  # vector of the average face
    normalized_A = A - A_face_vector  # average face subtracted from each image in A

    """Original SVD of Data Matrix"""
    SVD = npl.svd(normalized_A, full_matrices=False)
    U = SVD[0]
    S = SVD[1]
    Vh = SVD[2]

    """Power Iteration to Find Largest """
    ATA = np.matmul(normalized_A.T, normalized_A)
    PA = power_iteration(ATA)

    """SVD of the Data Matrix with Various Number of Randomized Samples"""
    U20, S20, Vh20 = randomized_SVD(normalized_A, 20)
    U50, S50, Vh50 = randomized_SVD(normalized_A, 50)
    U100, S100, Vh100 = randomized_SVD(normalized_A, 100)
    U250, S250, Vh250 = randomized_SVD(normalized_A, 250)
    U500, S500, Vh500 = randomized_SVD(normalized_A, 500)

    print("----------------Leading SVD Mode----------------")
    print(Vh[0,:])
    print("----------------Leading Singular Value----------------")
    print(S[0])
    print("----------------Leading Eigenvector from Power Iteration----------------")
    print(PA[1])
    print("----------------Sqrt of Leading Eigenvalue from Power Iteration----------------")
    print(np.sqrt(PA[0]))
    print("The eigenvector and eigenvalue squared obtained from power iteration matches the leading SVD mode and singular value.")


    """
    Comparing UXXX from Randomized SVD to U from Traditional SVD
    
    We looked at UTUXXX = np.matmul(U.T, UXXX), where UXXX is U20, U50, U100, U250, U500. 
    The closer they are, the closer UTUXXX will be to the identity matrix.
    
    It seems like no matter how many randomized samples you use, the modes do not seem to match up after about the first
    20%. 
    """
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.suptitle("Comparing U500 and U")
    pos1 = ax1.imshow(np.matmul(U.T, U500))
    pos2 = ax2.imshow(np.matmul(U.T, U500)[0:500,0:500])
    fig1.colorbar(pos1, ax=ax1)
    fig1.colorbar(pos2, ax=ax2)

    fig2, (ax3, ax4) = plt.subplots(1, 2)
    fig2.suptitle("Comparing U250 and U")
    pos3 = ax3.imshow(np.matmul(U.T, U250))
    pos4 = ax4.imshow(np.matmul(U.T, U250)[0:250, 0:250])
    fig2.colorbar(pos3, ax=ax3)
    fig2.colorbar(pos4, ax=ax4)

    fig3, (ax5, ax6) = plt.subplots(1, 2)
    fig3.suptitle("Comparing U100 and U")
    pos5 = ax5.imshow(np.matmul(U.T, U100))
    pos6 = ax6.imshow(np.matmul(U.T, U100)[0:100, 0:100])
    fig3.colorbar(pos5, ax=ax5)
    fig3.colorbar(pos6, ax=ax6)

    fig4, (ax7, ax8) = plt.subplots(1, 2)
    fig4.suptitle("Comparing U50 and U")
    pos7 = ax7.imshow(np.matmul(U.T, U50))
    pos8 = ax8.imshow(np.matmul(U.T, U50)[0:50, 0:50])
    fig4.colorbar(pos7, ax=ax7)
    fig4.colorbar(pos8, ax=ax8)

    fig5, (ax9, ax10) = plt.subplots(1, 2)
    fig5.suptitle("Comparing U20 and U")
    pos9 = ax9.imshow(np.matmul(U.T, U20))
    pos10 = ax10.imshow(np.matmul(U.T, U20)[0:20, 0:20])
    fig5.colorbar(pos9, ax=ax9)
    fig5.colorbar(pos10, ax=ax10)

    """
    Comparing VhXXX from Randomized SVD to Vh from Traditional SVD
    
    We looked at VhXXXVhT = np.matmul(VXXX, Vh.T), where VhXXX is Vh20, Vh50, Vh100, Vh250, Vh500. 
    The closer they are, the closer VhXXXVhT will be to the identity matrix.
    
    It seems like no matter how many randomized samples you use, the modes do not seem to match up after about the first
    20%.
    """

    fig6, (ax11, ax12) = plt.subplots(1, 2)
    fig6.suptitle("Comparing Vh500 and Vj")
    pos11 = ax11.imshow(np.matmul(Vh500, Vh.T).T)
    pos12 = ax12.imshow(np.matmul(Vh500, Vh.T)[0:500, 0:500])
    fig6.colorbar(pos11, ax=ax11)
    fig6.colorbar(pos12, ax=ax12)

    fig7, (ax13, ax14) = plt.subplots(1, 2)
    fig7.suptitle("Comparing Vh250 and Vj")
    pos13 = ax13.imshow(np.matmul(Vh250, Vh.T).T)
    pos14 = ax14.imshow(np.matmul(Vh250, Vh.T)[0:250, 0:250])
    fig7.colorbar(pos13, ax=ax13)
    fig7.colorbar(pos14, ax=ax14)

    fig8, (ax15, ax16) = plt.subplots(1, 2)
    fig8.suptitle("Comparing Vh100 and Vj")
    pos15 = ax15.imshow(np.matmul(Vh100, Vh.T).T)
    pos16 = ax16.imshow(np.matmul(Vh100, Vh.T)[0:100, 0:100])
    fig8.colorbar(pos15, ax=ax15)
    fig8.colorbar(pos16, ax=ax16)

    fig9, (ax17, ax18) = plt.subplots(1, 2)
    fig9.suptitle("Comparing Vh50 and Vj")
    pos17 = ax17.imshow(np.matmul(Vh50, Vh.T).T)
    pos18 = ax18.imshow(np.matmul(Vh50, Vh.T)[0:50, 0:50])
    fig9.colorbar(pos17, ax=ax17)
    fig9.colorbar(pos18, ax=ax18)

    fig10, (ax19, ax20) = plt.subplots(1, 2)
    fig10.suptitle("Comparing Vh20 and Vj")
    pos19 = ax19.imshow(np.matmul(Vh20, Vh.T).T)
    pos20 = ax20.imshow(np.matmul(Vh20, Vh.T)[0:20, 0:20])
    fig10.colorbar(pos19, ax=ax19)
    fig10.colorbar(pos20, ax=ax20)

    """
    Singular value decay for various numbers of randomized samples.
    """
    figscatter = plt.figure()
    figscatter.suptitle("Singular Value Decay for Various Numbers of Randomized Samples ")
    ax21 = figscatter.add_subplot(111)
    ax21.scatter(np.linspace(1,500,500), S500, s=10, alpha=0.25)
    ax21.scatter(np.linspace(1,500,500), S[0:500], s=10, alpha=0.25)
    ax21.scatter(np.linspace(1, 250, 250), S250, s=10, alpha=0.25)
    ax21.scatter(np.linspace(1, 100, 100), S100, s=10, alpha=0.25)
    ax21.scatter(np.linspace(1, 50, 50), S50, s=10, alpha=0.25)
    ax21.scatter(np.linspace(1, 20, 20), S20, s=10, alpha=0.25)

    plt.show()


