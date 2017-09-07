import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from cluster.clustering import get_labels


def tsne(num_clusters, vectors_to_visualise):
    model = TSNE(n_components=3, random_state=0, perplexity=20, learning_rate=1, n_iter=10000)
    np.set_printoptions(suppress=True)
    v = model.fit_transform(vectors_to_visualise)
    fig = plt.figure("Vectors in " + str(num_clusters) + " clusters")
    ax = Axes3D(fig)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_clusters-1, clip=True)
    mapper = cm.ScalarMappable(norm=norm)
    colors = list(map(mapper.to_rgba, get_labels(num_clusters, vectors_to_visualise)))
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], "o", c=colors)
    plt.show()


if __name__ == '__main__':
    tsne(int(input("Number of clusters: ")), [[1, 1, 1, 1], [5, 5, 4, 10], [15, 20, 0, 4]])
