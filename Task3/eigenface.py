import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread



def preprocess(A_pp):
    A = None
    Q_norms = None
    A_means = None


    A_pp_mean = np.mean(A_pp, axis=1, keepdims=True)
    Q = A_pp - A_pp_mean
    Q_norms = np.amax(np.abs(Q), axis=1, keepdims=True)
    A = Q / np.where(Q_norms == 0, 1, Q_norms)
    A_means = np.mean(A_pp, axis=1, keepdims=True)

    return A, Q_norms, A_means



def eigen_faces(A_pp):
    C = None
    F = None
    D = None
    Q_norms = None
    A_means = None


    A, Q_norms, A_means = preprocess(A_pp)
    D, C = np.linalg.eig(A.T @ A)
    D = D.real
    C = C.real


    F = A @ C
    F_norm = np.linalg.norm(F, axis=0, keepdims=True)
    F = F / F_norm

    C = A.T @ F
    return A, C, F, D, Q_norms, A_means



def reconstruct_image(Img, F, Q_norms, A_means,A):
    R = np.zeros((231, 195))


    #     calculate C, which is AT@A's eigenvector matrix
    C = A.T @ F

    #     calculate R_vector
    R_vector = np.zeros(Q_norms.shape)

    for j in range(len(Img)):
        R_vector = R_vector + (Img[j] * Q_norms * F[:, j].reshape(-1, 1))

    R_vector += A_means
    R = R_vector.reshape((231, 195), order="F")

    return R



def reduce_dimensionality(image_vector, k, F, D, A_means, Q_norms):
    compressed_image = None
    p = None


    #     k:k paras int
    #     F:eigen Face， A@A.T   eigenvector 45045,135
    #     D: A.T@A eigenvalue 135,
    #     A_means: average face lookings 45045,1
    #     Q_norms: image normalized       45045,1
    #     image_vector: pic we need to compress ,45045,1

    #     intialize image_vector

    Q = image_vector.reshape(-1, 1) - A_means
    image_normalized = Q / np.where(Q_norms == 0, 1, Q_norms)

    #     handle compressed image
    F_k = F[:, 0:k]

    compressed_parameter = F_k.T @ image_normalized
    compressed_parameter_s = np.zeros((135, 1))
    for i in range(135):
        if (i < k):
            compressed_parameter_s[i] = compressed_parameter[i]


    #     handle p
    d_sum = np.sum(D)
    fenzi = 0
    for i in range(k):
        fenzi += D[i]
    p = fenzi / d_sum

    print(image_normalized.shape)
    print(compressed_parameter.shape)
    print(F_k.shape)


    return compressed_parameter_s, p



if __name__ == '__main__':
    # load images
    yale_path = os.path.join("Yale-FaceA", "trainingset")
    yale_images = []
    image_num = 0
    for image_name in os.listdir(yale_path):
        image_num+=1
        image_path = os.path.join(yale_path,image_name)
        im = imread(image_path)
        im = im.flatten('F')  # flatten im into a vector
        yale_images.append(im)
    yale_images = np.stack(yale_images).T  # build a matrix where each column is a flattened image

    print(f"# of yale images: {len(yale_images)}")
    print("image shape 231 X 195")
    print(f"yale_images shape: {yale_images.shape}")
    # plt.imshow(yale_images[:,0].reshape((231, 195),order='F'),cmap='gray')
    # plt.show()

    # preprocess
    # A, Q_norms, A_means = preprocess(yale_images)
    # print(A.shape)
    # print(Q_norms.shape)
    # print(A_means.shape)

    # eigenface
    A, C, F, D, Q_norms, A_means = eigen_faces(yale_images)
    print('Orthogonality Check (should be close to 0): ', F[:, 0].T @ F[:, 1])
    print('Unit Vector Check: ', math.isclose(F[:, 0].T @ F[:, 0], 1))
    # print(C.shape)
    # print(F.shape)


    # MEAN FACE ANSWER
    mean_face = A_means.reshape((231, 195),order='F')
    print(F.shape)
    print(C.shape)
    # plt.imshow(mean_face,cmap='gray')
    # plt.show()

    Idx = 133
    Img = (A[:, Idx]).T @ F
    R = reconstruct_image(Img, F, Q_norms, A_means,A)
    # plt.imshow(R,cmap='gray')
    print(np.amax(R), np.amin(R))
    # plt.show()


    # R = np.zeros((231, 195))
    # for c in range(195):
    #     R[:, c] = (yale_images[:, Idx])[c * 231: (c + 1) * 231]
    # plt.imshow(R,cmap='gray')
    # print(np.amax(R), np.amin(R))
    # plt.show()

    compressed_image, p = reduce_dimensionality(yale_images[:, Idx], 10, F, D, A_means, Q_norms)

    print('Variance Captured:', int(p * 100), '%')

    R_c = reconstruct_image(compressed_image, F, Q_norms, A_means,A)
    plt.imshow(R_c,cmap='gray')
    print(np.amax(R_c), np.amin(R_c))
    plt.show()
    R_o = reconstruct_image(Img, F, Q_norms, A_means,A)
    plt.imshow(R_o,cmap='gray')
    print(np.amax(R_o), np.amin(R_o))
    plt.show()
    print('Error : \n', np.sum(np.abs(R_c - R_o)))

