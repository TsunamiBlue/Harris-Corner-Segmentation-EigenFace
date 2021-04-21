"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Jieli Zheng (u6579712)
"""

import numpy as np
import cv2


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01

k=0.01

nms_size = 11


# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

bw_original = cv2.imread('Harris_1.jpg')
bw_original = cv2.cvtColor(bw_original,cv2.COLOR_BGR2RGB)
bw = cv2.cvtColor(bw_original,cv2.COLOR_RGB2GRAY)
# bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################
# print(bw.shape)
# print(Ix2.shape)
# print(Iy2.shape)
# print(Ixy.shape)


def harris_corner(bw, Ix2, Iy2, Ixy, k=0.01):
    """
    harris corner detect algorithm w/o nms operations and thresholding.
    second order derivatives along x and y axis are expected.
    :param bw: the original grayscale image.
    :param Ix2: the second order derivative of the image w.r.t x axis.
    :param Iy2: the second order derivative of the image w.r.t y axis.
    :param Ixy: the second order derivative of the image w.r.t both x and y
    :param k: the empirical constant for MCR. default 0.01
    :return: measure of corner response for all pixels in the image.
    """
    # measure of corner response
    R = np.zeros(bw.shape)
    # loop through all pixels
    for i, row in enumerate(bw):
        for j, pixel in enumerate(row):
            # M =[[Ix^2, Ixy] , [Ixy, Iy^2]], and its determinant and trace are shown below:
            current_determinant = Ix2[i][j]*Iy2[i][j]-2*Ixy[i][j]
            current_trace = Ix2[i][j] + Iy2[i][j]
            # measure of corner response = det(M) - k(trace(M))^2
            current_MCR = current_determinant - k*np.power(current_trace,2)
            # R > threshold to be recognized as a corner.
            # print(current_MCR.shape)
            R[i][j] = current_MCR

    return R



R = harris_corner(bw, Ix2, Iy2, Ixy,k=k)

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################


def nms_thresholding(R, nms_size=3, d=0.01):
    """
    Non-maximum Suppression a.w.a thresholding for MCR
    :param R: measure of corner response for a certain image.
    :param nms_size: decide how large the nms window is. odd number is expected.
    :param d: thresholding. default 0.01
    :return: the result map of the image.
    """
    # the result map of the image
    ans = []
    # shift from centre pixel.
    shift = int((nms_size-1)/2)
    threshold = 0.01*np.max(R)
    for i, row in enumerate(R):
        for j, mcr in enumerate(row):
            # firstly, mcr should pass the threshold.
            if mcr > threshold:
                # slice the window.
                upward_idx = max(i-shift,0)
                downward_idx = min(i+shift,R.shape[0])
                left_idx = max(j-shift,0)
                right_idx = min(j+shift,R.shape[1])
                # gather neighbour MCRs
                nms_neighbours = np.array(R[upward_idx:downward_idx+1, left_idx:right_idx+1]).reshape(-1,1)

                # if it's local maximum, we recognize it as a real corner.
                if (mcr >= nms_neighbours).all():
                    ans.append([j,i])

    return ans


ans= nms_thresholding(R,nms_size=nms_size,d=thresh)


sample_ans = cv2.cornerHarris(bw,nms_size,3,k)
sample_ans = nms_thresholding(sample_ans,nms_size=nms_size,d=thresh)

ans = np.array(ans)
sample_ans = np.array(sample_ans)
print(f"Corner detected: {ans.shape[0]} vs {sample_ans.shape[0]}")



# plotting
plt.figure(num='Harris Corner Detection',figsize=(12,12))

plt.subplot(1,2,1)
plt.title(f"result w/ nms size={nms_size}")
plt.imshow(bw_original)
plt.plot(ans[:,0],ans[:,1],'rx')

plt.subplot(1,2,2)
plt.title(f"sample result w/ nms size={nms_size}")
plt.imshow(bw_original)
plt.plot(sample_ans[:,0],sample_ans[:,1],'rx')
plt.show()
