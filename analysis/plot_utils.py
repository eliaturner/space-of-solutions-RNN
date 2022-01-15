import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def prepare_plot_delay_graph(ax, high_ts):
    x = range(0, high_ts+1)
    ax.plot(x, x, zorder=1, color='grey')
    rect = patches.Rectangle((30, 30), 91, 91, linewidth=0.5, edgecolor='grey', facecolor='grey', alpha=0.1)
    # rect = patches.Rectangle((0, 0), 240, 90, linewidth=1, edgecolor='grey', facecolor='grey',
    #                          alpha=0.1)

    ax.add_patch(rect)
    # ax.set_ylim(bottom=100, top=200)
    # ax.set_xlim(left=100, right=200)
    ax.set_xlabel(r'$t_s$', fontsize=14)
    ax.set_ylabel(r'$t_p$', fontsize=14)
    # ax.set_title(r'$t_s$ vs. $t_p$', fontsize=22)


class PCATransformer:
    def __init__(self, dim):
        self.scaler = StandardScaler()
        self.pca = PCA(dim, svd_solver='full')
        self.dim = dim

    def fit_transform(self, X):
        X = X.reshape(-1, X.shape[-1])
        # X = self.scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X

    def fit(self, X):
        self.fit_transform(X)
        return

    def transform(self, X):
        X = X.reshape(-1, X.shape[-1])
        # X = self.scaler.transform(X)
        X = self.pca.transform(X)
        X = self.scaler.transform(X)
        return X


def plot_n_states(s_lists, labels=None, dim=2, colors=None):
    pca = PCA(dim)
    pca.fit(np.vstack(s_lists))
    fig = plt.figure()
    ax = plt.axes(projection='3d' if dim == 3 else None)
    from matplotlib import cm
    if colors is None:
        colors = cm.Reds(np.linspace(0.1, 0.9, len(s_lists)))

    if labels is None:
        labels = len(s_lists)*['']
    for i, s in enumerate(s_lists):
        s_low2 = pca.transform(s)
        if dim == 3:
            ax.plot3D(s_low2[:,0], s_low2[:,1], s_low2[:,2],  color=colors[i], linewidth=3)
        else:
            ax.plot(*np.hsplit(pca.transform(s), dim), label=labels[i], color=colors[i], linewidth=3)
            # ax.scatter(*np.hsplit(pca.transform(s), dim), color='white', edgecolors='black', zorder=3, s=50)

        ax.scatter(*np.hsplit(pca.transform(s[:1]), dim), s=50, zorder=0, color=colors[i])
        # ax.scatter(*np.hsplit(pca.transform(s[-1:]), dim), color='black', marker='x', s=50, zorder=0)
    plt.axis('off')
    return pca
    # plt.legend()

    #plt.show()

def scatter_n_states(s_lists, labels=None):
    pca = PCA(2)
    pca.fit(np.vstack(s_lists))
    plt.clf()
    if labels is None:
        labels = len(s_lists)*['']
    for i, s in enumerate(s_lists):
        plt.scatter(*np.hsplit(pca.transform(s), 2), label=labels[i])
        plt.scatter(*np.hsplit(pca.transform(s[-1:]), 2), color='black', marker='x')
    # plt.show()
    plt.axis('off')

    plt.legend()