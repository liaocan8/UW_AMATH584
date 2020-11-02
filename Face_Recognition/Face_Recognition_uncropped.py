from skimage.io import imread
import numpy as np
import numpy.linalg as npl
import matplotlib.pylab as plt
import os as oss
import os.path as op
from random import randint

def dir_list(folder):
    """

    :param folder: A folder containing files
    :return: A list of strings where the strings are Python readable file names of the files in folder
    """
    top = r'C:\Users\liaoc\Desktop\Library_of_Knowledge\UW classes\AMATH 584 Applied Linear Algebra\hw\hw 2 data'
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
        list_of_images.append(imread(i, as_grey=True))
    return list_of_images

def average_face(imagelist):
    """

    :param imagelist: list of images
    :return: averaged image of images in the folder
    """
    a_face = sum(imagelist)/len(imagelist)
    return a_face

if __name__ == '__main__':
    all_images = image_list('yalefaces')
    im_shape = all_images[0].shape # shape of an image

    A = np.array([i.flatten() for i in all_images]).T # data matrix where each column is an image
    A_face_vector = np.array([np.sum(A,axis=1)/len(all_images)]).T # vector of the average face
    normalized_A = A - A_face_vector # average face subtracted from each image in A

    SVD = npl.svd(normalized_A, full_matrices=False)
    U = SVD[0]
    S = SVD[1]
    Vh = SVD[2]
    SVh = np.matmul(np.diag(S), Vh)
    """
    Does SVD and returns (U, S, V*)
    Since data is inputted into A as columns:
    U is the matrix with eigenfaces as its columns in order of highest to lowest singular value
    S is a vector of singular values: higher the singular value, more "common" the eigenface is
    Vh has information about how each image projects onto the eigenspace:
        ith column of SVh are coefficients for re-creating the ith training image as a linear combination of eigenfaces
    """

    R = randint(0, len(all_images)-1)

    recon_training0_5modes = np.matmul(U[:, 0:5], (SVh[0:5, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_10modes = np.matmul(U[:, 0:10], (SVh[0:10, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_15modes = np.matmul(U[:, 0:15], (SVh[0:15, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_25modes = np.matmul(U[:, 0:25], (SVh[0:25, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_50modes = np.matmul(U[:, 0:50], (SVh[0:50, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_75modes = np.matmul(U[:, 0:75], (SVh[0:75, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_100modes = np.matmul(U[:, 0:100], (SVh[0:100, R])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_150modes = np.matmul(U[:,0:150],(SVh[0:150,R])).reshape(im_shape) + A_face_vector.reshape(im_shape)

    fig = plt.figure(f"Reconstruction of Training Image {R} using Different Sized Eigenface Spaces", figsize=(20, 20))
    im0 = fig.add_subplot(191)
    im1 = fig.add_subplot(192)
    im2 = fig.add_subplot(193)
    im3 = fig.add_subplot(194)
    im4 = fig.add_subplot(195)
    im5 = fig.add_subplot(196)
    im6 = fig.add_subplot(197)
    im7 = fig.add_subplot(198)
    OG = fig.add_subplot(199)

    im0.imshow(recon_training0_5modes)
    im0.set_title("5 Modes")

    im1.imshow(recon_training0_10modes)
    im1.set_title("10 Modes")

    im2.imshow(recon_training0_15modes)
    im2.set_title("15 Modes")

    im3.imshow(recon_training0_25modes)
    im3.set_title("25 Modes")

    im4.imshow(recon_training0_50modes)
    im4.set_title("50 Modes")

    im5.imshow(recon_training0_75modes)
    im5.set_title("75 Modes")

    im6.imshow(recon_training0_100modes)
    im6.set_title("100 Modes")

    im7.imshow(recon_training0_150modes)
    im7.set_title("150 Modes")

    OG.imshow(A[:, R].reshape(im_shape))
    OG.set_title("Original")
    """
    We will try to reconstruct Training Image R as a linear combination of different number of eigenfaces to see about 
    how big our basis needs to be.
    """

    plt.figure("Figure 2: Singular Value Spectrum", figsize=(10, 10))
    plt.title("Figure 2: Singular Value Spectrum")
    plt.scatter(range(1, len(S) + 1), S)
    """
    We plot the singular values in hierarchical order to see which eigenfaces are needed to classify a new image to a
    person.
    """

    plt.show()