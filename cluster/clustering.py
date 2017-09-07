from bisect import insort
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def cluster(num_clusters, vectors_to_cluster):
    # vectors_to_cluster = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    k_means = KMeans(init='k-means++', n_clusters=num_clusters)
    x = StandardScaler().fit_transform(vectors_to_cluster)
    k_means.fit(x)
    clusters = {}
    for count, cluster_id in enumerate(k_means.labels_):
        if cluster_id in clusters:
            insort(clusters[cluster_id], vectors_to_cluster[count])
        else:
            clusters[cluster_id] = [vectors_to_cluster[count]]

    for cluster_id in range(num_clusters):
        print("Cluster ", cluster_id, clusters[cluster_id])
    print("\n\n")
    return clusters


def get_labels(num_clusters, vectors_to_cluster):
    # vectors_to_cluster = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    k_means = KMeans(init='k-means++', n_clusters=num_clusters)
    k_means.fit(vectors_to_cluster)
    return k_means.labels_
