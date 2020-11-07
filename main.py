import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from math import sqrt
from random import randint
from scipy.io import arff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from time import time


# KMeans #

def _kmeans(n_clusters, data):
    return KMeans(n_clusters=n_clusters, init='k-means++').fit_predict(data)


# Agglomerative Clustering #

def _agglomerative_clustering(n_clusters, data, linkage='ward'):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(data)


# DBSCAN #

def _dbscan(data, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)


def compute_eps(xy, k):
    sum_dis = 0
    sample_size = 50
    for i0 in range(sample_size):
        i1 = randint(0, len(xy) - 1)
        min_dis = []
        for i2 in range(0, len(xy)):
            if i1 == i2:
                continue
            dis = sqrt((xy[i1][0] - xy[i2][0]) ** 2 + (xy[i1][1] - xy[i2][1]) ** 2)
            for i3 in range(k):
                if len(min_dis) < k:
                    min_dis.append(dis)
                    if len(min_dis) == k:
                        min_dis.sort()
                    break
                elif dis < min_dis[i3]:
                    min_dis[i3] = dis
                    break
        sum_dis += min_dis[k - 1]
    return round(sum_dis / sample_size, 4)


def calculate_nb_clusters(clustering, min_cluster_size=50):
    cluster_count = [0] * len(set(clustering))
    for i in clustering:
        if i != -1:
            cluster_count[i] += 1
    size = 0
    for i in range(len(cluster_count)):
        if cluster_count[i] >= min_cluster_size:
            size += 1
    return size


def _compute_dbscan(data, compute_method, n_clusters=0, default_clusters=2):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    xy = [[d[0], d[1]] for d in data]

    if n_clusters != 0:
        best_clustering = compute_method(xy)
        plt.xlabel('Clusters : %d' % len(set(best_clustering)), fontsize=8)
    else:
        min_samples = 2
        best = -1
        best_clustering = []
        best_min_samples = -1
        best_eps = -1
        start = time()
        while min_samples <= 15:
            eps = compute_eps(xy, min_samples)
            clustering = compute_method(xy, eps, min_samples)
            if len(set(clustering)) == 1:
                min_samples += 1
                continue
            score = silhouette_score(xy, clustering, metric='euclidean')
            if score > best:
                best = score
                best_clustering = clustering
                best_min_samples = min_samples
                best_eps = eps
            min_samples += 1
        plt.xlabel('Clusters : %d \n Score : %.2f (eps : %.2f - min_samples : %d) \n Duration : %.2fs' % (calculate_nb_clusters(best_clustering), best, best_eps, best_min_samples, round(time() - start, 3)),
                   fontsize=8)

    plt.scatter(x, y, c=best_clustering, s=point_size)


# HDBSCAN #

def _hdbscan(data, min_samples=5):
    return HDBSCAN(min_samples=min_samples, min_cluster_size=50).fit_predict(data)


def _compute_hdbscan(data, compute_method, n_clusters=0, default_clusters=2):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    xy = [[d[0], d[1]] for d in data]

    if n_clusters != 0:
        best_clustering = compute_method(xy)
        plt.xlabel('Clusters : %d' % len(set(best_clustering)), fontsize=8)
    else:
        min_samples = 2
        best = -1
        best_clustering = []
        best_min_samples = 2
        start = time()
        while min_samples <= 15:
            clustering = compute_method(xy, min_samples)
            if len(set(clustering)) == 1:
                min_samples += 1
                continue
            score = silhouette_score(xy, clustering, metric='euclidean')
            if score > best:
                best = score
                best_clustering = clustering
                best_min_samples = min_samples
            min_samples += 1
        plt.xlabel('Clusters : %d \n Score : %.2f (min_samples : %d) \n Duration : %.2fs' % (len(set(best_clustering)), best, best_min_samples, round(time() - start, 3)),
                   fontsize=8)

    plt.scatter(x, y, c=best_clustering, s=point_size)


# General #

def compute_clustering(data, compute_method, n_clusters=0, default_clusters=2):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    xy = [[d[0], d[1]] for d in data]
    if n_clusters == 0:
        clustering = compute_n_clusters(xy, compute_method, default_clusters)
    else:
        clustering = compute_method(n_clusters, xy)
    plt.scatter(x, y, c=clustering, s=point_size)


def compute_n_clusters(data, compute_method, n_clusters=2):
    max_clusters = n_clusters + 10
    best = -1
    best_clustering = []
    nb_clusters = 2
    start = time()
    while n_clusters <= max_clusters:
        clustering = compute_method(n_clusters, data)
        score = silhouette_score(data, clustering, metric='euclidean')
        if score > best:
            best = score
            best_clustering = clustering
            nb_clusters = n_clusters
        n_clusters += 1
    plt.xlabel('Clusters : %d \n Score : %.2f \n Duration : %.2fs' % (nb_clusters, best, round(time() - start, 3)),
               fontsize=8)
    return best_clustering


def load_dataset(filename, folder='artificial', extension='arff'):
    return arff.loadarff(open('datasets/' + folder + '/' + filename + '.' + extension, 'r'))


def launch_clustering(dataset, compute_method, method_name, compute_clustering_method=compute_clustering):
    print('Start to compute ' + method_name)
    plt.figure(num=method_name, figsize=(16, 8), dpi=80)
    plot_id = 241
    for data in dataset:
        plt.subplot(plot_id)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlabel('Clusters : %d' % data[1], fontsize=8)
        if plot_id == 241:
            plt.ylabel('n_clusters known', fontsize=14, fontweight='bold')
        compute_clustering_method(data[0][0], compute_method, data[1])

        plt.subplot(plot_id+4)
        plt.xticks([], [])
        plt.yticks([], [])
        if plot_id == 241:
            plt.ylabel('n_clusters unknown', fontsize=14, fontweight='bold')
        compute_clustering_method(data[0][0], compute_method, default_clusters=max(data[1]-5, 2))

        plot_id += 1


# Main #

point_size = 1

if __name__ == '__main__':
    data1 = load_dataset('2d-10c')
    data2 = load_dataset('2d-3c-no123')
    data3 = load_dataset('2d-4c')
    data4 = load_dataset('banana')

    #dataset = [(data1, 9), (data2, 3), (data3, 4), (data4, 2)]

    data1 = load_dataset('2d-10c')
    data2 = load_dataset('diamond9')
    data3 = load_dataset('complex9')
    data4 = load_dataset('banana')

    #dataset = [(data1, 9), (data2, 9), (data3, 9), (data4, 2)]

    data1 = load_dataset('dense-disk-3000')
    data2 = load_dataset('compound')
    data3 = load_dataset('ds3c3sc6')
    data4 = load_dataset('complex8')

    dataset = [(data1, 2), (data2, 2), (data3, 2), (data4, 8)]

    data1 = load_dataset('x1', 'random', 'txt')
    data2 = load_dataset('x2', 'random', 'txt')
    data3 = load_dataset('x3', 'random', 'txt')
    data4 = load_dataset('x4', 'random', 'txt')

    #dataset = [(data1, 15), (data2, 15), (data3, 15), (data4, 15)]

    '''
    launch_clustering(
        dataset,
        _kmeans,
        'KMeans'
    )
    launch_clustering(
        dataset,
        _agglomerative_clustering,
        'Agglomerative Clustering - Ward'
    )
    launch_clustering(
        dataset,
        lambda n_clusters, data: _agglomerative_clustering(n_clusters, data, 'complete'),
        'Agglomerative Clustering - Complete'
    )
    launch_clustering(
        dataset,
        lambda n_clusters, data: _agglomerative_clustering(n_clusters, data, 'average'),
        'Agglomerative Clustering - Average'
    )
    launch_clustering(
        dataset,
        lambda n_clusters, data: _agglomerative_clustering(n_clusters, data, 'single'),
        'Agglomerative Clustering - Single'
    )
    '''
    launch_clustering(
        dataset,
        _dbscan,
        'DBSCAN',
        _compute_dbscan
    )
    '''
    launch_clustering(
        dataset,
        _hdbscan,
        'HDBSCAN',
        _compute_hdbscan
    )
    '''

    plt.show()
