import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools.utils import load_pickle


def plot_cluster_histogram(sizes, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    sizes = sorted(sizes)[::-1]
    limit = np.argwhere(np.array(sizes) < 10).squeeze()[0]
    num_taken = sum(sizes[:limit])
    percentage = int(100*num_taken/400)
    labels = len(sizes)*['']

    # plt.clf()
    plt.bar(np.arange(len(sizes)), sizes)
    plt.xticks(np.arange(len(sizes)), labels)
    plt.axvline(limit - 0.5)
    plt.axhline(10, linestyle='--', color='grey')
    ax.text(limit + 1, 80, rf'${percentage}\%$', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.set_facecolor('white')

    plt.grid(False)
    # plt.show()
    plt.xlabel('Reduced-Dynamics type', size=20)
    plt.ylabel('count', size=20)
    plt.savefig(f'figures/hist_{name}.pdf')
    plt.close()
    #plt.show()

def flatten_list(l):
    l_f = []
    for e in l:
        if type(e) == tuple:
            l_f.append(e[0])
            l_f.append(e[1])
        else:
            l_f.append(e)

    return np.array(l_f)


def scale(X):
    return StandardScaler().fit_transform(X)


def return_dict_sorted_by_size(d):
    l = [(key, len(d[key])) for key in d.keys()]
    l = sorted(l, key=lambda p : p[1])[::-1]
    l = [a[0] for a in l]
    d_new = {}
    for i, graph in enumerate(l):
        d_new[i] = graph

    return d_new


def rectify_data(X, y):
    count = np.bincount(y)
    idx = np.argwhere(count >= 10).squeeze()
    all_indices = np.sort(np.concatenate([np.argwhere(y == i).squeeze() for i in idx]))
    return X[all_indices], y[all_indices]
def predict_class_from_training_features(task, architecture):
    y = load_pickle(f'files/y_{task}_{architecture}')
    features = load_pickle(f'files/features_{task}_{architecture}')
    to_remove = np.argwhere(y).squeeze()
    features = features[to_remove]
    y = y[to_remove]
    y = np.array(y - 1, dtype=int)

    X = scale(features)
    iterations = 50

    X, y = rectify_data(X, y)
    confusion_matrices = []
    scores = []
    counts = np.bincount(np.array(y, dtype=int))
    n_clusters = len(np.unique(y))
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        a = confusion_matrix(y_test, y_pred, normalize='true', labels=np.arange(n_clusters))
        confusion_matrices.append(a)
        scores.append(cohen_kappa_score(y_test, y_pred))

    confusion_matrices = np.stack(confusion_matrices)
    mean_mat = np.mean(confusion_matrices, axis=0)
    scores = np.array(scores)
    fig, ax = plt.subplots(1, 1)
    for (i, j), z in np.ndenumerate(mean_mat):
        if max(i, j) >= len(counts) or counts[i] < 10 or counts[j] < 10:
            mean_mat[i, j] = 0
            ax.text(j, i, 'N/A', ha='center', va='center', size=10)
        else:
            pass
    p = ax.imshow(mean_mat, cmap='binary', vmin=0, vmax=1)

    print(r'$\overline{\kappa}=$'+rf'${np.round(np.mean(scores), 2)}$')
    plt.axis('OFF')
    plt.savefig(f'figures/matrix_{task}_{architecture}.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    predict_class_from_training_features('delayed_discrimination', 'vanilla')
    predict_class_from_training_features('interval_discrimination', 'vanilla')
    predict_class_from_training_features('time_reproduction', 'vanilla')
    predict_class_from_training_features('time_reproduction', 'gru')
    predict_class_from_training_features('time_reproduction', 'lstm')




