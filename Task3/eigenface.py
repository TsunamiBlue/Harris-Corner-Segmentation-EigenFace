import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# TODO
# A_pp is images.
A_pp = None


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


# A, Q_norms, A_means = preprocess(A_pp)
# print(A.shape)
# print(Q_norms)
# print(A_means)

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

    #     C=C.astype(np.float64)
    F = A @ C
    F_norm = np.linalg.norm(F, axis=0, keepdims=True)
    F = F / F_norm
    #     D=D.astype(np.float64)
    C = A.T @ F
    return C, F, D, Q_norms, A_means


# # For the purposes of doing this assignment, this code isn't really here. Pretend it's engraved in rock.
# C, F, D, Q_norms, A_means = eigen_faces(A_pp)
# print('Orthogonality Check (should be close to 0): ', F[:, 0].T @ F[:, 1])
# print('Unit Vector Check: ', math.isclose(F[:, 0].T @ F[:, 0], 1))
# print(C.shape)
# print(F.shape)


def reconstruct_image(Img, F, Q_norms, A_means):
    R = np.zeros((243, 320))


    #     calculate C, which is AT@A's eigenvector matrix
    C = A.T @ F

    #     calculate R_vector
    R_vector = np.zeros(Q_norms.shape)

    for j in range(len(Img)):
        R_vector = R_vector + (Img[j] * Q_norms * F[:, j].reshape(-1, 1))

    R_vector += A_means
    R = R_vector.reshape((243, 320), order="F")

    return R



def reduce_dimensionality(image_vector, k, F, D, A_means, Q_norms):
    compressed_image = None
    p = None


    #     k:k paras int
    #     F:eigen Face， A@A.T   eigenvector 77760,165
    #     D: A.T@A eigenvalue 165,
    #     A_means: average face lookings 77760,1
    #     Q_norms: image normalized       77760,1
    #     image_vector: pic we need to compress ,77760,1

    #     intialize image_vector

    Q = image_vector.reshape(-1, 1) - A_means
    image_normalized = Q / np.where(Q_norms == 0, 1, Q_norms)

    #     handle compressed image
    F_k = F[:, 0:k]

    compressed_parameter = F_k.T @ image_normalized
    compressed_parameter_s = np.zeros((165, 1))
    for i in range(165):
        if (i < k):
            compressed_parameter_s[i] = compressed_parameter[i]

    #     compressed_parameter=np.zeros((165,1))
    #     for i in range(k):
    #         compressed_parameter[i]=F_k.T[i]@image_vector

    #     handle p
    d_sum = np.sum(D)
    fenzi = 0
    for i in range(k):
        fenzi += D[i]
    p = fenzi / d_sum

    print(image_normalized.shape)
    print(compressed_parameter.shape)
    print(F_k.shape)
    #     compressed_image=A_means+F_k@compressed_parameter
    #     print(compressed_image.shape)

    return compressed_parameter_s, p


# # Display Code. Leave it alooooooooooone.
# # You can mess with settings, but return them to their original values.
# compressed_image, p = reduce_dimensionality(A_pp[:, Idx], 10, F, D, A_means, Q_norms)
#
# print('Variance Captured:', int(p * 100), '%')
#
# R_c = reconstruct_image(compressed_image, F, Q_norms, A_means)
# plt.imshow(R_c)
# print(np.amax(R_c), np.amin(R_c))
# plt.show()
# R_o = reconstruct_image(Img, F, Q_norms, A_means)
# plt.imshow(R_o)
# print(np.amax(R_o), np.amin(R_o))
# plt.show()
# print('Error (expect around 1700000 for k = 10, 0 for k = 165): \n', np.sum(np.abs(R_c - R_o)))


if __name__ == '__main__':
    # load images
    yale_path = os.path.join("Yale-FaceA", "trainingset")
    yale_images = []
    for image_name in os.listdir(yale_path):
        image_path = os.path.join(yale_path,image_name)
        image = cv2.imread(image_path)
        yale_images.append(image)
    print(f"yale images: {len(yale_images)}")
    yale_images = np.array(yale_images)
    print(yale_images.shape)


