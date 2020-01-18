"""
Created by Alex Wang
On 2018-11-30
"""
import traceback

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

def test_pca():
    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)

    mean = pca.mean_
    components = pca.components_
    X = X - mean
    X = np.dot(X, components.T)
    print('type of mean:{}, type of component:{}'.format(type(mean), type(components)))

    # X = pca.transform(X)

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def test_ipca():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    n_components = 2
    ipca = decomposition.IncrementalPCA(n_components=n_components, batch_size=10)
    X_ipca_org = ipca.fit_transform(X)

    mean = ipca.mean_
    components = ipca.components_
    X_ipca = X - mean
    X_ipca = np.dot( X_ipca, components.T)


    pca = decomposition.PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    colors = ['navy', 'turquoise', 'darkorange']

    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        plt.figure(figsize=(8, 8))
        for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
            plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                        color=color, lw=2, label=target_name)

        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of iris dataset\nMean absolute unsigned error "
                              "%.6f" % err)
        else:
            plt.title(title + " of iris dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.axis([-4, 4, -1.5, 1.5])

    plt.show()


if __name__ == '__main__':
    # test_pca()
    test_ipca()