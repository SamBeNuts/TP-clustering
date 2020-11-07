import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from math import sqrt
from random import randint
from scipy.io import arff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def calculate_kn_distance(xy, k):
    kn_distance = []
    for i in range(len(xy)):
        eucl_dist = []
        for j in range(len(xy)):
            eucl_dist.append(sqrt(((xy[i][0] - xy[j][0]) ** 2) + ((xy[i][1] - xy[j][1]) ** 2)))
        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])
    return kn_distance


def compute_eps(xy, k):
    all_min_dis = []
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
        all_min_dis.append(min_dis[k - 1])
    all_min_dis.sort()
    return round(sum(all_min_dis[:50]) / 50, 4)


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


if __name__ == '__main__':
    data = arff.loadarff(open('datasets/artificial/complex9.arff', 'r'))
    x = [d[0] for d in data[0]]
    y = [d[1] for d in data[0]]
    xy = [[d[0], d[1]] for d in data[0]]
    # eps = compute_eps(xy, 4)
    # print(eps)
    # eps_dist = calculate_kn_distance(xy, 4)
    # plt.hist(eps_dist, bins=30)
    # plt.ylabel('n')
    # plt.xlabel('Epsilon distance')
    plt.figure(figsize=(14, 7))
    for i in range(2, 12):
        plt.subplot(2, 5, i-1)
        eps = compute_eps(xy, i)
        print(eps)
        clustering = DBSCAN(eps=eps, min_samples=i).fit_predict(xy)
        # clustering = HDBSCAN(min_samples=i, min_cluster_size=50).fit_predict(xy)
        plt.scatter(x, y, c=clustering, s=1)
        #nb_clusters = len(set(clustering))
        nb_clusters = calculate_nb_clusters(clustering)
        if nb_clusters > 1:
            # xy_clean = []
            # clustering_clean = []
            # for j in range(len(xy)):
            #     if clustering[j] != -1:
            #         xy_clean.append(xy[j])
            #         clustering_clean.append(clustering[j])
            score = silhouette_score(xy, clustering, metric='euclidean')
            plt.xlabel('Clusters : %d - Score : %.2f' % (nb_clusters, score), fontsize=8)
    plt.show()
