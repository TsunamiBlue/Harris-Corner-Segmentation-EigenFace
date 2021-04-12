import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


# K mean clustering & its helper methods

# k means ++ initialization
def kmeanspp(image,centroid_num):
    X = image.reshape(-1,1)
    m = centroid_num

    ans = []
    fst_index = np.random.randint(0, len(X))
    ans.append(X[fst_index])
    d = [0 for _ in range(len(X))]
    for _ in range(1, m):
        sum1 = float(0)
        for i, point in enumerate(X):
            _, d[i] = nearest_centroid(ans, point)
            sum1 += d[i]
        sum1 *= np.random.random()
        for i, a in enumerate(d):
            sum1 -= a
            if (sum1 > 0):
                continue
            if (check_duplicate(X[i], ans)):
                continue
            ans.append(X[i])
            break

    return np.array(ans)


def check_duplicate(sample, arrs):
    for arr in arrs:
        if np.array_equal(sample, arr):
            return True
    return False


def nearest_centroid(centroids,point):
    distance_min = np.inf
    ans_idx = -2
    for i,centroid in enumerate(centroids):
        distance = float(np.linalg.norm(centroid-point))
        if distance < distance_min:
            ans_idx = i
            distance_min=distance
    # which centroid it belongs to
    return ans_idx, distance_min


def my_kmeans(image, centroid_num, using_coords=True, using_kmeanspp=False,iteration_num=10):
    """
    my k-mean clustering algorithm as well as some adv args
    :param image: np array of 3 channels
    :param centroid_num: number of centroids, default 6
    :param using_coords: whether to consider the coords of the pixel when clustering
    :param using_kmeanspp: whether to use k-means++ when initialization
    :param iteration_num: number of iterations, default 10
    :return: final clusters
    """
    clusters = dict()
    image_shape = image.shape
    assignments = np.zeros(image_shape[:2])
    assignments -= 2
    # initialize
    if using_kmeanspp:
        if using_coords:
            images_coords = []
            for i in range(image_shape[0]):
                tmp = []
                for j in range(image_shape[1]):
                    tmp.append([i,j])
                images_coords.append(tmp)

            images_coords = np.array(images_coords)
            images_coords = images_coords.transpose(2,0,1)
            data_points = np.vstack([image.transpose(2,0,1),images_coords])
            data_points = data_points.transpose(1,2,0)
            print(data_points.shape)
            centroids = kmeanspp(data_points,centroid_num)
        else:
            centroids = kmeanspp(image,centroid_num)
    else:
        centroids = np.random.rand(centroid_num,3)
        centroids *= 255
        centroids = centroids.astype(np.uint8)

        centroids_coords = np.random.rand(centroid_num,2)
        centroids_coords[:,0] *=image_shape[0]
        centroids_coords[:,1] *= image_shape[1]
        centroids_coords = centroids_coords.astype(np.uint8)

        if using_coords:
            centroids = np.hstack((centroids,centroids_coords))


    # main iterations:
    for i in range(iteration_num):

        clusters = dict()
        for m in range(centroid_num):
            clusters[m] = []

        # E - step
        for j, row in enumerate(image):
            for k, channels in enumerate(row):
                if using_coords:
                    point = np.hstack((channels,j,k))
                else:
                    point = channels
                current_idx,_ = nearest_centroid(centroids,point)
                assignments[j][k] = current_idx
                # record assignments
                clusters[current_idx].append(point)

        # M - step
        # re-generate new centroids by the mean of points in cluster
        new_centroids = []
        for centroid_idx, points in clusters.items():
            if points:
                current_new_centroid = np.mean(points, axis=0)
            else:
                # remain
                current_new_centroid = centroids[centroid_idx]
            new_centroids.append(current_new_centroid)
        centroids = np.array(new_centroids)

    if using_coords:
        return clusters
    else:
        return clusters,assignments



if __name__ == '__main__':
    # load image
    mandm_pth = os.path.join("mandm.png")
    peppers_pth = os.path.join("peppers.png")

    mandm_image = cv2.imread(mandm_pth)
    mandm_image = cv2.cvtColor(mandm_image, cv2.COLOR_BGR2LAB)
    print(f"mandm size {mandm_image.shape}")

    # cluster with coords and without kmeans++
    ans_clusters = my_kmeans(mandm_image,7,using_kmeanspp=True,using_coords=True)
    # visualize
    clustered_image = np.zeros(mandm_image.shape)
    clustered_image -= 1
    clustered_image = clustered_image.astype(np.uint8)
    for centroid_idx, points in ans_clusters.items():
        print(len(points))
        if points:
            print(np.array(points).shape)
            centroid_mean = np.mean(points,axis=0)
            L,a,b,_,_ = centroid_mean
            centroid_Lab = np.array([L, a, b]).astype(np.uint8)
            for _,_,_,x,y in points:
                clustered_image[x][y] = centroid_Lab

    clustered_image = cv2.cvtColor(clustered_image,cv2.COLOR_LAB2RGB)
    plt.imshow(clustered_image)
    plt.show()


    # # visualize
    # ans_clusters_no_coord, ans_assignments = my_kmeans(mandm_image, 7, using_kmeanspp=False, using_coords=False)
    # print(ans_assignments)
    # clustered_image_no_coord = np.zeros(mandm_image.shape)
    # clustered_image_no_coord -= 1
    # clustered_image = clustered_image_no_coord.astype(np.uint8)
    # # calculate the mean colour of each cluster
    # cluster_mean_colour = dict()
    # for centroid_idx, points in ans_clusters_no_coord.items():
    #     if points:
    #         centroid_mean = np.mean(points,axis=0)
    #         centroid_Lab = np.array(centroid_mean).astype(np.uint8)
    #         cluster_mean_colour[centroid_idx] = centroid_Lab
    #
    # for x,row in enumerate(ans_assignments):
    #     for y,cluster_idx in enumerate(row):
    #         clustered_image_no_coord[x][y] = cluster_mean_colour[cluster_idx]
    #
    # clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_LAB2RGB)
    # plt.imshow(clustered_image)
    # plt.show()

