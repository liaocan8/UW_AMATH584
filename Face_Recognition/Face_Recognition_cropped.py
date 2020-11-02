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

if __name__ == '__main__':
    images_by_folder = [image_list(i) for i in [f'yaleB{n}' for n in range(1,39)]]
    # makes a list of lists of images where the sub-list is a list of images in yaleB{n}
    all_images = [] # all training images
    for i in images_by_folder:
        for j in i:
            all_images.append(j)
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
    A_face_vector.reshape(im_shape)
    R = randint(0,38)

    recon_training0_5modes = np.matmul(U[:, 0:5], (SVh[0:5, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_10modes = np.matmul(U[:, 0:10], (SVh[0:10, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_25modes = np.matmul(U[:, 0:25], (SVh[0:25, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_50modes = np.matmul(U[:, 0:50], (SVh[0:50, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_100modes = np.matmul(U[:, 0:100], (SVh[0:100, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_500modes = np.matmul(U[:, 0:500], (SVh[0:500, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_1000modes = np.matmul(U[:, 0:1000], (SVh[0:1000, R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)
    recon_training0_2000modes = np.matmul(U[:,0:2000],(SVh[0:2000,R*64])).reshape(im_shape) + A_face_vector.reshape(im_shape)

    fig = plt.figure(f"Reconstruction of Training Image {R*64} using Different Sized Eigenface Spaces", figsize=(20, 20))
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

    im2.imshow(recon_training0_25modes)
    im2.set_title("25 Modes")

    im3.imshow(recon_training0_50modes)
    im3.set_title("50 Modes")

    im4.imshow(recon_training0_100modes)
    im4.set_title("100 Modes")

    im5.imshow(recon_training0_500modes)
    im5.set_title("500 Modes")

    im6.imshow(recon_training0_1000modes)
    im6.set_title("1000 Modes")

    im7.imshow(recon_training0_2000modes)
    im7.set_title("2000 Modes")

    OG.imshow(A[:, R*64].reshape(im_shape))
    OG.set_title("Original")

    """
    We will try to reconstruct Training Image R as a linear combination of different number of eigenfaces to see about 
    how big our basis needs to be.
    """

    eigen5_vector = U[:,4]
    eigen6_vector = U[:,5]

    eigen9_vector = U[:, 8]
    eigen10_vector = U[:, 9]

    eigen24_vector = U[:, 23]
    eigen25_vector = U[:, 24]

    eigen49_vector = U[:, 48]
    eigen50_vector = U[:, 49]

    eigen99_vector = U[:, 98]
    eigen100_vector = U[:, 99]

    eigen499_vector = U[:, 498]
    eigen500_vector = U[:, 499]

    eigen999_vector = U[:, 998]
    eigen1000_vector = U[:, 999]

    eigen1999_vector = U[:, 1998]
    eigen2000_vector = U[:, 1999]

    x = randint(0,37)
    y = randint(0,37)

    rando1 = images_by_folder[x]
    rando2 = images_by_folder[y]

    dot_rando1_eigen5 = [np.dot(i.flatten(), eigen5_vector) for i in rando1]
    dot_rando1_eigen6 = [np.dot(i.flatten(), eigen6_vector) for i in rando1]
    dot_rando2_eigen5 = [np.dot(i.flatten(), eigen5_vector) for i in rando2]
    dot_rando2_eigen6 = [np.dot(i.flatten(), eigen6_vector) for i in rando2]

    dot_rando1_eigen9 = [np.dot(i.flatten(), eigen9_vector) for i in rando1]
    dot_rando1_eigen10 = [np.dot(i.flatten(), eigen10_vector) for i in rando1]
    dot_rando2_eigen9 = [np.dot(i.flatten(), eigen9_vector) for i in rando2]
    dot_rando2_eigen10 = [np.dot(i.flatten(), eigen10_vector) for i in rando2]

    dot_rando1_eigen24 = [np.dot(i.flatten(), eigen24_vector) for i in rando1]
    dot_rando1_eigen25 = [np.dot(i.flatten(), eigen25_vector) for i in rando1]
    dot_rando2_eigen24 = [np.dot(i.flatten(), eigen24_vector) for i in rando2]
    dot_rando2_eigen25 = [np.dot(i.flatten(), eigen25_vector) for i in rando2]

    dot_rando1_eigen49 = [np.dot(i.flatten(), eigen49_vector) for i in rando1]
    dot_rando1_eigen50 = [np.dot(i.flatten(), eigen50_vector) for i in rando1]
    dot_rando2_eigen49 = [np.dot(i.flatten(), eigen49_vector) for i in rando2]
    dot_rando2_eigen50 = [np.dot(i.flatten(), eigen50_vector) for i in rando2]

    dot_rando1_eigen99 = [np.dot(i.flatten(), eigen99_vector) for i in rando1]
    dot_rando1_eigen100 = [np.dot(i.flatten(), eigen100_vector) for i in rando1]
    dot_rando2_eigen99 = [np.dot(i.flatten(), eigen99_vector) for i in rando2]
    dot_rando2_eigen100 = [np.dot(i.flatten(), eigen100_vector) for i in rando2]

    dot_rando1_eigen499 = [np.dot(i.flatten(), eigen499_vector) for i in rando1]
    dot_rando1_eigen500 = [np.dot(i.flatten(), eigen500_vector) for i in rando1]
    dot_rando2_eigen499 = [np.dot(i.flatten(), eigen499_vector) for i in rando2]
    dot_rando2_eigen500 = [np.dot(i.flatten(), eigen500_vector) for i in rando2]

    dot_rando1_eigen999 = [np.dot(i.flatten(), eigen999_vector) for i in rando1]
    dot_rando1_eigen1000 = [np.dot(i.flatten(), eigen1000_vector) for i in rando1]
    dot_rando2_eigen999 = [np.dot(i.flatten(), eigen999_vector) for i in rando2]
    dot_rando2_eigen1000 = [np.dot(i.flatten(), eigen1000_vector) for i in rando2]

    dot_rando1_eigen1999 = [np.dot(i.flatten(), eigen1999_vector) for i in rando1]
    dot_rando1_eigen2000 = [np.dot(i.flatten(), eigen2000_vector) for i in rando1]
    dot_rando2_eigen1999 = [np.dot(i.flatten(), eigen1999_vector) for i in rando2]
    dot_rando2_eigen2000 = [np.dot(i.flatten(), eigen2000_vector) for i in rando2]

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 5 vs. Projection onto Eigenface 6")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 5 vs. Projection onto Eigenface 6")
    plt.ylabel("Projection onto Eigenvector 5")
    plt.xlabel("Projection onto Eigenvector 6")
    plt.scatter(dot_rando1_eigen6, dot_rando1_eigen5)
    plt.scatter(dot_rando2_eigen6, dot_rando2_eigen5)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 9 vs. Projection onto Eigenface 10")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 9 vs. Projection onto Eigenface 10")
    plt.ylabel("Projection onto Eigenvector 9")
    plt.xlabel("Projection onto Eigenvector 10")
    plt.scatter(dot_rando1_eigen10, dot_rando1_eigen9)
    plt.scatter(dot_rando2_eigen10, dot_rando2_eigen9)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 24 vs. Projection onto Eigenface 25")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 24 vs. Projection onto Eigenface 25")
    plt.ylabel("Projection onto Eigenvector 24")
    plt.xlabel("Projection onto Eigenvector 25")
    plt.scatter(dot_rando1_eigen25, dot_rando1_eigen24)
    plt.scatter(dot_rando2_eigen25, dot_rando2_eigen24)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 49 vs. Projection onto Eigenface 50")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 49 vs. Projection onto Eigenface 50")
    plt.ylabel("Projection onto Eigenvector 49")
    plt.xlabel("Projection onto Eigenvector 50")
    plt.scatter(dot_rando1_eigen50, dot_rando1_eigen49)
    plt.scatter(dot_rando2_eigen50, dot_rando2_eigen49)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 99 vs. Projection onto Eigenface 100")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 99 vs. Projection onto Eigenface 100")
    plt.ylabel("Projection onto Eigenvector 99")
    plt.xlabel("Projection onto Eigenvector 100")
    plt.scatter(dot_rando1_eigen100, dot_rando1_eigen99)
    plt.scatter(dot_rando2_eigen100, dot_rando2_eigen99)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 499 vs. Projection onto Eigenface 500")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 499 vs. Projection onto Eigenface 500")
    plt.ylabel("Projection onto Eigenvector 499")
    plt.xlabel("Projection onto Eigenvector 500")
    plt.scatter(dot_rando1_eigen500, dot_rando1_eigen499)
    plt.scatter(dot_rando2_eigen500, dot_rando2_eigen499)

    plt.figure(f"Training Person {x+1} and {y+1} Projection onto Eigenface 999 vs. Projection onto Eigenface 1000")
    plt.title(f"Training Person {x+1} and {y+1} Projection onto Eigenface 999 vs. Projection onto Eigenface 1000")
    plt.ylabel("Projection onto Eigenvector 999")
    plt.xlabel("Projection onto Eigenvector 1000")
    plt.scatter(dot_rando1_eigen1000, dot_rando1_eigen999)
    plt.scatter(dot_rando2_eigen1000, dot_rando2_eigen999)

    plt.figure(f"Projection onto Eigenface 1999 vs. Projection onto Eigenface 2000")
    plt.title(f"Projection onto Eigenface 1999 vs. Projection onto Eigenface 2000")
    plt.ylabel("Projection onto Eigenvector 1999")
    plt.xlabel("Projection onto Eigenvector 2000")
    plt.scatter(dot_rando1_eigen2000, dot_rando1_eigen1999)
    plt.scatter(dot_rando2_eigen2000, dot_rando2_eigen1999)
    """
    To see what number of eigenfaces are needed to distinguish between faces, plot (x,y) where x is the projection
    of a picture of a random person's image onto eigenface n and y is the projection onto eigenface n-1.
    """

    plt.figure("Figure 2: Singular Value Spectrum", figsize=(10,10))
    plt.title("Figure 2: Singular Value Spectrum")
    plt.scatter(range(1,len(S)+1),S)
    """
    We plot the singular values in hierarchical order to see which eigenfaces are needed to classify a new image to a
    person.
    """

    plt.show()