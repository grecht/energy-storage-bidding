import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def det_prob(labels, k):
    counts = np.zeros(k)
    for i in prange(len(labels)):
        counts[labels[i]] += 1
    return counts / len(labels)


@jit(nopython=True)
def kmeans(X, k):
    centroids = k_means_plus_plus(X, k)

    labels = closest_centroid(X, centroids)
    centroids = det_new_centroids(X, centroids, labels)

    # 300 iterations, matching scikit-learn
    for _ in range(300):
        new_labels = closest_centroid(X, centroids)
        new_centroids = det_new_centroids(X, centroids, new_labels)

        if np.array_equal(new_labels, labels):
            return labels, centroids
        labels = new_labels
        centroids = new_centroids
    return labels, centroids


# https://louisabraham.github.io/articles/broadcasting-and-numba.html
@jit(nopython=True, parallel=True)
def closest_centroid(data, centroids):
    n, d = data.shape
    k, _ = centroids.shape
    answer = np.empty(n, dtype=np.int64)
    for i in prange(n):
        min_dist_i = np.finfo(np.float64).max
        for j in range(k):
            dist_i_j = 0
            for u in range(d):
                dist_i_j += (data[i, u] - centroids[j, u]) ** 2
            if dist_i_j < min_dist_i:
                min_dist_i = dist_i_j
                answer[i] = j
    return answer


@jit(nopython=True, parallel=True)
def det_new_centroids(X, centroids, closest):
    new_centroids = np.zeros(centroids.shape)
    for k in prange(len(centroids)):
        vals = X[closest==k]
        new_centroids[k] = (1/len(vals)) * vals.sum(axis=0)
    return new_centroids


@jit(nopython=True)
def k_means_plus_plus(X, k):
    # https://github.com/hjafk/Kmeans-pp-numpy-implementation
    m, num_dims = X.shape
    centroids = np.zeros((k, num_dims), dtype=np.float64)

    centroids[0, :] = X[np.random.choice(m), :]
    num_centroids = 1

    while num_centroids < k:
        distance = np.zeros((m, centroids.shape[0]), dtype=np.float64)  # init distance
        for i in range(centroids.shape[0]):
            distance[:, i] = np.sum(np.square(X - centroids[i, :]), axis=1)

        total_distance = 0
        for i in range(distance.shape[0]):
            total_distance += np.min(distance[i, :])

        prob = np.zeros(distance.shape[0], dtype=np.float64)
        for i in range(distance.shape[0]):
            prob[i] = np.min(distance[i, :]) / total_distance

        centroids[num_centroids, :] = X[rand_ind_choice(prob)]
        num_centroids += 1
    return centroids

@jit(nopython=True)
def rand_ind_choice(pmf):
    # https://github.com/numba/numba/issues/2539#issuecomment-507306369
    return np.searchsorted(np.cumsum(pmf), np.random.random(), side="right")