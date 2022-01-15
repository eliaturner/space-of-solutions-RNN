import numpy as np


def transition_indices(input, output):
    if output is None:
        seq = input
    else:
        seq = np.concatenate([input, output], axis=-1)
    n_time = input.shape[0]
    transitions = [i + 1 for i in range(n_time - 1) if (seq[i] != seq[i + 1]).any()]
    # transitions = [i for i in transitions if np.max(seq[i]) in [0, 1.]]
    transitions = [0] + transitions + [n_time]
    return transitions


def get_rsg_idx(x, y):
    one_idx = np.argwhere(x[-1,:,0] == 1).flatten()
    ready = one_idx[0]
    if one_idx.size == 20:
        set = one_idx[10]
    else:
        set = np.argwhere(x[-1,:,1] == 1).flatten()[0]

    go = np.argwhere(y[-1] == 1).flatten()[0]

    return ready, set, go


def extrapolation_score(ts_tp, threshold=10, fraction=0):
    for ts in range(30, 121):
        if ts not in ts_tp:
            return 0

    score = 30
    total = failure = 0
    for key, val in ts_tp.items():
        if key - score > 2:
            break
        total += 1
        if abs(key - val) > threshold:
            failure += 1
            if failure / total > fraction:
                break

        else:
            score = key

    return score
