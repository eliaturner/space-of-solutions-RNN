import numpy as np


def set_root(e, index, root):
    # Set all nodes to point to a new root.
    while e[index] < index:
        e[index], index = root, e[index]
    e[index] = root


def find_root(e, index):
    # Find the root of the tree from node index.
    root = index
    while e[root] < root:
        root = e[root]
    return root


def union(e, i, j):
    # Combine two trees containing node i and j.
    # Return the root of the union.
    root = find_root(e, i)
    if i != j:
        root_j = find_root(e, j)
        if root > root_j:
            root = root_j
        set_root(e, j, root)
    set_root(e, i, root)
    return root


def flatten_label(e):
    # Flatten the Union-Find tree and relabel the components.
    label = 1
    for i in range(1, len(e)):
        if e[i] < i:
            e[i] = e[e[i]]
        else:
            e[i] = label
            label += 1


def scan(a, width, height):  # 4-connected
    l = [[0 for _ in range(width)] for _ in range(height)]
    p = [0]  # Parent array.
    label = 1
    # Assumption: 'a' has been padded with zeroes (bottom and right parts
    # does not require padding).
    for y in range(1, height):
        for x in range(1, width):
            if a[y][x] == 0:
                continue
            # Decision tree for 4-connectivity.
            if a[y - 1][x]:  # b
                if a[y][x - 1]:  # d
                    l[y][x] = union(p, l[y - 1][x], l[y][x - 1])
                else:
                    l[y][x] = l[y - 1][x]
            elif a[y][x - 1]:  # d
                l[y][x] = l[y][x - 1]
            else:
                # new label
                l[y][x] = label
                p.append(label)
                label += 1
    return l, p


def get_entry(a, i, j):
    if i == 1:
        return None, None
    for m in range(1, 4):
        if a[i - m, j]:  # b
            return i - m, j

    if j <= 30 + i:
        return None, None

    row_offset = 15
    col_offset_l = min(50, j - 35 - i)
    col_offset_r = 50
    col_offset_r = 20
    col_offset_l = min(col_offset_r, j - 35 - i)
    sub_a = a[i - row_offset:i, j-col_offset_l:j+col_offset_r]
    rows, cols = np.where(sub_a > 0)
    if rows.size > 0:
        return rows[-1] + (i-row_offset), cols[-1] + (j-col_offset_l)

    return None, None




def scan(a):
    width = a.shape[1]
    height = a.shape[0]
    l = np.zeros((height, width), dtype=int)
    p = [0]
    label = 1
    # Assumption: 'a' has been padded with zeroes (bottom and right parts
    # does not require padding).
    for y in range(1, height):
        for x in range(1, width):
            if a[y, x] == 0:
                continue

            if x in [y + 30, y + 31]:
                continue

            row, col = get_entry(a, y, x)
            if row is not None:
                if a[y, x - 1]:  # d
                    l[y, x] = union(p, l[row, col], l[y, x - 1])
                else:
                    l[y, x] = l[row, col]
            else:
                if a[y, x - 1]:  # d
                    l[y, x] = l[y, x - 1]
                else:
                    # new label
                    l[y, x] = label
                    p.append(label)
                    label += 1



    flatten_label(p)
    for y in range(len(l)):
        for x in range(len(l[0])):
            l[y][x] = p[l[y][x]]

    p = np.unique(p)
    for c in range(1, p.size - 1):
        i0, i1 = np.where(l == c)
        i0, i1 = i0[-1], i1[-1]
        j0, j1 = np.where(l == c + 1)
        j0, j1 = j0[0], j1[0]
        print(i0, i1)
        print(j0, j1)

    return l, p

