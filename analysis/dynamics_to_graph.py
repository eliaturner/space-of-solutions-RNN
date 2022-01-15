import seaborn as sns

from analysis.analyzer import Analyzer
from tools.math_utils import calc_normalized_q_value

DISTANCE_THRESHOLD = 1
OUTPUT_THRESHOLD = 0.01
sns.set()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
from analysis.process_graph import get_zero_paths, get_path_from_source
from analysis.process_graph import plot_interval_reproduction_graph, make_compact, plot_interval_discrimination_graph, plot_delayed_discrimination_graph
from analysis.graph_utils import *
from tools.math_utils import svm_if_separable


def find_merge_indices(states1, outputs1, states2, outputs2):
    mat = states_distance_matrix(states2, states1)
    mat_output = max_bottom_left_diagonal(distance_matrix(outputs2, outputs1))
    path2_idx, path1_idx, min_val = calculate_merge_index(mat)
    while mat_output[path2_idx, path1_idx] > 0.1:
        path1_idx, path2_idx = path1_idx + 1, path2_idx + 1
        if path2_idx == len(states2) or path1_idx == len(states1):
            raise NameError

    return path1_idx, path2_idx

def check_if_and_where_components_merging_is_possible(c1, c2):
    paths_c1 = get_zero_paths(c1)[0]
    paths_c2 = get_zero_paths(c2)[0]

    compatible_paths = {}
    for idx1, path1 in enumerate(paths_c1):
        path1_states, path1_outputs = get_state_output_from_path(c1, path1)
        compatible_paths[idx1] = None
        min_index = np.inf
        for idx2, path2 in enumerate(paths_c2):
            path2_states, path2_outputs = get_state_output_from_path(c2, path2)
            try:
                path1_idx, path2_idx = find_merge_indices(path1_states, path1_outputs, path2_states, path2_outputs)
                if path1_idx < min_index:
                    compatible_paths[idx1] = (len(path1) - path1_idx, idx2, len(path2) - path2_idx)
                    min_index = path1_idx

            except NameError:
                continue

    triplets = list(compatible_paths.values())
    merge_size = [min(t[0], t[2]) if t is not None else 0 for t in triplets]
    best_path1 = np.argmax(merge_size)
    if merge_size[best_path1] > 0:
        t = triplets[best_path1]
        node1 = paths_c1[best_path1][-t[0]]
        node2 = paths_c2[t[1]][-t[2]]
        return node1, node2

    return -1, -1


def merge_two_paths(G, path1, path2):
    for j in range(min(len(path2), len(path1))):
        G = nx.contracted_nodes(G, path1[j], path2[j])

    s = path1[j]
    targets_zero = sorted([t for t in G.successors(s) if G[s][t]['edge_type'] == 0])
    while len(targets_zero) > 1:
        G = nx.contracted_nodes(G, targets_zero[0], targets_zero[1])
        s = targets_zero[0]
        targets_zero = sorted([t for t in G.successors(s) if G[s][t]['edge_type'] == 0])

    return G


def merge_components(G):
    while True:
        G_zero = get_zero_graph(G)
        components = [nx.subgraph(G_zero, c) for c in nx.weakly_connected_components(G_zero)]
        pairs = [(c1, c2) for c1 in components for c2 in components if c1 != c2]
        for (c1, c2) in pairs:
            node1, node2 = check_if_and_where_components_merging_is_possible(c1, c2)
            if node1 != -1:
                path1 = get_path_from_source(c1, node1)
                path2 = get_path_from_source(c2, node2)
                G = merge_two_paths(G, path1, path2)
                break

        else:
            return G

class DynamicsExceptionAbstract(Exception):
    pass


class DynamicsContainCycle(DynamicsExceptionAbstract):
    def __init__(self, dynamics, outputs, t, s):
        self.dynamics = dynamics
        self.outputs = outputs
        self.t = t
        self.s = s


class DynamicsNameLater(DynamicsExceptionAbstract):
    def __init__(self, dynamics, outputs):
        self.dynamics = dynamics
        self.outputs = outputs


class WeirdTrajectory(DynamicsExceptionAbstract):
    def __init__(self, dynamics, outputs):
        self.dynamics = dynamics
        self.outputs = outputs

def generate_input(n, steps):
    input = np.zeros((n, steps, 2))
    input[:,:5, 1] = 1
    return input


def distance_diagonal(a, TS):
    a_norm = np.linalg.norm(a[:-1] - a[1:], axis=1)
    a_norm = np.append(a_norm, a_norm[-1])
    for k in TS:
        if k <= 0:
            continue
        for i in range(max(len(a)-200, 0), len(a) - k):
            for j in range(i + k, len(a), k):
                d = np.linalg.norm(a[i] - a[j])
                if d < 1e-5 or (d < 1 and 2*d/(a_norm[i] + a_norm[j]) < 1 and abs(np.log10(a_norm[i]) - np.log10(a_norm[j])) < 1):
                    continue
                else:
                    break
            else:
                return (i + k, i)

    raise NameError


def max_bottom_left_diagonal(mat):
    n, m = mat.shape
    for i in range(n - 2, -1, -1):
        for j in range(m - 2, -1, -1):
            mat[i, j] = max(mat[i, j], mat[i + 1, j + 1])

    return mat

def speed_vector(vec):
    vec_norm = np.linalg.norm(vec[:-1] - vec[1:], axis=1)
    vec_norm = np.append(vec_norm, vec_norm[-1])
    return vec_norm

def states_distance_matrix(a, b):
    a_norm, b_norm = speed_vector(a), speed_vector(b)
    n, m = len(a), len(b)
    mat = 1000*np.ones((n, m))
    for k in range(-n-1, m):
        for j in range(m - 1, max(k - 1,-1), -1):
            i = j - k
            if i >= n:
                continue

            d = np.linalg.norm(a[i] - b[j])
            if d > 1e-3:
                mat[i, j] = 2*d/(a_norm[i] + b_norm[j])
                if mat[i, j] > 5 or abs(np.log10(a_norm[i]+ 1e-10) - np.log10(b_norm[j] + 1e-10)) > 1:
                    break
            else:
                mat[i, j] = 0

    mat = max_bottom_left_diagonal(mat)
    return mat

def diagonal_index_to_matrix_index(location, k):
    if k >= 0:
        row, col = location, location + k
    else:
        row, col = location - k, location

    return row, col


def calculate_merge_index(mat):
    if np.min(mat) >= 5:
        raise NameError('A very specific bad thing happened')

    n, m = mat.shape[0], mat.shape[1]
    diagonals = []
    min_val = np.inf
    for j in range(m):
        min_row = np.argmin(mat[:,j])
        if mat[min_row, j] < DISTANCE_THRESHOLD:
            min_val = min(mat[min_row, j], min_val)
            diagonals.append(j - min_row)
            if min_val == 0:
                break

    #different for extrapolation vs training, where you don't have long inputs
    from collections import Counter
    most_common_2 = Counter(diagonals).most_common(2)

    if most_common_2 and (most_common_2[0][1] > 3 or min_val < 0.01 or abs(n-m) < 5):
        if len(most_common_2) == 2 and most_common_2[0][1] > 2*most_common_2[1][1]:
            k = most_common_2[0][0]
        else:
            k = diagonals[-1]
    else:
        raise NameError('A very specific bad thing happened')

    diagonal = np.array(np.diag(mat, k=k))
    location = np.argwhere(diagonal < DISTANCE_THRESHOLD).flatten()[0]
    row, col = diagonal_index_to_matrix_index(location, k)
    return row, col, min_val


def merge_paths(G, path_s, path_t):
    path_s = path_s[:len(path_t)]
    for idx in range(len(path_t) - 1, -1, -1):
        G = nx.contracted_nodes(G, path_s[idx], path_t[idx])

    return G

def get_merging_location(G, lace_states, lace_outputs, exclude_node):
    paths = get_zero_paths(G)
    compatible_paths = []#{}

    for component in paths.keys():
        for path in paths[component]:
            if exclude_node == path[0]:
                continue
            path_states = get_property_from_path(G, path, 'state')
            path_outputs = get_property_from_path(G, path, 'output')
            try:
                mat = states_distance_matrix(path_states, lace_states);
                mat_output = max_bottom_left_diagonal(distance_matrix(path_outputs[:], lace_outputs[:], p=1))
                path_idx, lace_idx, min_val = calculate_merge_index(mat)
                while mat_output[path_idx, lace_idx] > OUTPUT_THRESHOLD:
                    path_idx += 1
                    lace_idx += 1
                    if path_idx == len(path_states) or lace_idx == len(lace_states):
                        raise NameError

                compatible_paths.append((lace_idx, path[path_idx:], min_val))

            except NameError:
                continue

    if not bool(compatible_paths):
        lace_offset, path = len(lace_states), []
    else:
        pairs = compatible_paths#list(compatible_paths.values())
        offsets = [p[0] for p in pairs]
        idx = np.argmin(offsets)
        lace_offset, path, _ = pairs[idx]

    return lace_offset, path

def merge_two_cycles(G, cycle1, cycle2):
    cycle1_states = get_property_from_path(G, cycle1, 'state')
    cycle2_states = get_property_from_path(G, cycle2, 'state')
    dists = [np.linalg.norm(cycle2_states[0] - cycle1_states[i]) for i in range(len(cycle1_states))]
    candidate, candidate_val = np.argmin(dists), np.min(dists)
    G.remove_nodes_from(cycle2[1:])
    G = nx.contracted_nodes(G, cycle1[candidate], cycle2[0])
    return G

def merge_cycle_to_graph(G, start_node):
    zero_graph = get_zero_graph(G)
    curr_cycle = [e[0] for e in nx.find_cycle(zero_graph, start_node)]
    curr_cycle_states = get_property_from_path(G, curr_cycle, 'state')
    node = curr_cycle[0]
    all_cycles = nx.simple_cycles(zero_graph)
    all_cycles = [cycle for cycle in all_cycles if node not in cycle and len(cycle) > 1]
    all_cycles = sorted(all_cycles, key=lambda x:x[0])
    all_cycles = [cycle for cycle in all_cycles if 0.8 <= len(cycle) / len(curr_cycle) <= 1.2]
    distance_threshold = 0.1 if len(curr_cycle) >= 10 else 0
    all_cycles = [cycle for cycle in all_cycles if
                  svm_if_separable(get_property_from_path(G, cycle, 'state'), curr_cycle_states) <= distance_threshold]
    if all_cycles:
        curr_cycle, all_cycles[0] = all_cycles[0], curr_cycle
        for cycle in all_cycles:
            G = merge_two_cycles(G, curr_cycle, cycle)

    return G



def try_merge_dynamics_to_graph(G, start_node):
    # TODO make sure start_node is start.
    lace_path = get_path_from_source(G, start_node, None)
    if get_target(G, lace_path[-1], 0):
        G = merge_cycle_to_graph(G, start_node)
        return G

    lace_states = get_property_from_path(G, lace_path, 'state')
    lace_outputs = get_property_from_path(G, lace_path, 'output')
    lace_offset, path = get_merging_location(G, lace_states, lace_outputs, start_node)
    set_node_attributes(G, path, lace_states[lace_offset:], 'state')
    set_node_attributes(G, path, lace_outputs[lace_offset:], 'output')
    G.remove_nodes_from(lace_path[lace_offset + 1:])
    if path:
        neighbors = [neigh for neigh in G[lace_path[lace_offset]]]
        for neigh in neighbors:
            G.remove_edge(lace_path[lace_offset], neigh)

        G = nx.contracted_nodes(G, path[0], lace_path[lace_offset], self_loops=False)
    return G


def clip_dynamics(dynamics, outputs):
    q_vals = calc_normalized_q_value(dynamics[None, :]).flatten()
    threshold = 5e-5

    if min(q_vals) < threshold:
        idx = np.argwhere(q_vals[:] < threshold).flatten()[0]
        dynamics = dynamics[:idx]
        outputs = outputs[:idx]
    elif len(dynamics) > 1000 and np.all(np.diff(q_vals[200:]) <= 0):
        dynamics = dynamics[:-1]
        outputs = outputs[:-1]
    return dynamics, outputs


def get_cycle(dynamics):
    dyn1d = PCA(1).fit_transform(dynamics).squeeze()
    corr = np.correlate(dyn1d, dyn1d, mode='full')[:len(dyn1d)]
    corr = corr/np.max(corr)
    corr[corr < 0] = 0
    from scipy.signal import find_peaks
    peaks = find_peaks(corr, prominence=0.05)[0]
    if peaks.size == 0:
        raise NameError
    if peaks.size < 6:
        return [len(dynamics) - 1 - peaks[0]]
    diffs = np.hstack([np.diff(peaks[:]), np.diff(peaks[::2])])
    bins = np.bincount(diffs)
    TS = np.argsort(bins)[-4:]
    TS = [T for T in TS if bins[T] > 1]
    return TS[::-1]


def locate_cycle(dynamics):
    TS = get_cycle(dynamics)
    t, s = distance_diagonal(dynamics, TS)
    return t, s


def connect_cycle(G, t, s):
    nodes_to_remove = np.arange(t, len(G))
    G.remove_nodes_from(nodes_to_remove)
    G.add_edge(t - 1, s)
    G[t-1][s]['edge_type'] = 0


def get_sink_node(G):
    sinks = (node for node in G.nodes() if G.out_degree(node) == 0)
    sink = next(sinks, None)
    if not sink:
        l = [node for node in G.nodes() if G.in_degree(node) > 1]
        if l:
            return l[0]

        *_, sink = G.nodes()
    return sink


def get_connected_components(G):
    connected_components = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
    return connected_components

def create_lace_from_dynamics(dynamics, outputs, try_locate_cycle = True):
    G = create_lace(dynamics, outputs)
    if not try_locate_cycle:
        return G
    try:
        t, s = locate_cycle(dynamics)
        connect_cycle(G, t, s)
    except:
        pass
    return G

def connect_lace_to_graph(G, lace, prev_node, edge_type):
    start_node = get_available_node(G)
    lace = shift_nodes_names(lace, start_node)
    G = nx.union(G, lace)
    if prev_node is not None:
        G.add_edge(prev_node, start_node)
        G[prev_node][start_node]['edge_type'] = edge_type

    return G, start_node

def try_connect_components(G, model):
    G_zero = get_zero_graph(G)
    connected_components = get_connected_components(G_zero)
    if len(connected_components) == 1:
        return G

    sinks = [get_sink_node(c) for c in connected_components]
    initial_states = [G.nodes[node]['state'] for node in sinks]
    initial_states = np.vstack(initial_states)
    pred = model.run_system_from_inits(initial_states, steps=10000)['state']
    mat = pairwise_distances(pred[:, -1])
    if np.max(mat) < 1e-3:
        for node in sinks[1:]:
            neighbors = [neigh for neigh in G_zero[node]]
            for neigh in neighbors:
                G.remove_edge(node, neigh)
            G = nx.contracted_nodes(G, sinks[0], node, self_loops=False)

    return G


def get_max_peak(array):
    from scipy.signal import find_peaks
    peaks = find_peaks(array, prominence=0.1)[0]
    if len(peaks):
        return peaks[0]

    return -1


def modified_cycle_finder(dynamics, delta):
    if len(dynamics) == delta:
        raise NameError
    a = distance_matrix(dynamics[delta:], dynamics[delta:])
    if len(dynamics) == 2*delta:
        min_diag = 2
    else:
        min_diag = 20

    # diags = [np.argmin(a[i, i+1:]) + 1 for i in range(0, len(a)-1)]
    diagonals = [get_max_peak(-a[i, i + 1:]) + 1 for i in range(len(a))]
    from collections import Counter
    most_commons = Counter(diagonals).most_common(10)
    most_commons = [pair for pair in most_commons if pair[0] > min_diag and pair[1] > 10]
    if most_commons:
        k, count = most_commons[0][0], most_commons[0][1]
        if (len(a) - k) <= 3 * count or len(a) > 400 and count > 30:
            offset = np.argwhere(np.array(diagonals) == k).squeeze()[0] + delta
            return k + offset, offset

    raise NameError


# TODO:EDIT
class DynamicsToGraphAbstract(Analyzer):
    name = 'dynamics_to_diagram_abstract'
    value_descriptor = 'none'

    @property
    def dynamics_steps_threshold(self):
        pass

    @property
    def dynamics_steps_delta(self):
        pass

    def generate_input1(self, n, steps):
        input = np.zeros((n, steps, 2))
        input[:, :self.data_params.pulse, 0] = 1
        return input

    def generate_input2(self, n, steps):
        input = np.zeros((n, steps, 2))
        input[:, :self.data_params.pulse, 1] = 1
        return input

    def get_dynamics(self, initial_state, steps):
        pred = self.model.run_system_from_inits(initial_state[None, :], steps=steps)
        return clip_dynamics(pred['state'][0], pred['output'])

    def create_lace_from_initial_state(self, initial_state):
        dynamics, outputs = self.get_dynamics(initial_state, self.dynamics_steps_threshold)
        G = create_lace(dynamics, outputs)
        try:
            t, s = modified_cycle_finder(dynamics, self.dynamics_steps_delta)
            connect_cycle(G, t, s)
            dynamics, outputs = dynamics[:len(G)], outputs[:len(G)]
        except:
            pass

        return G, dynamics, outputs


    def extend_graph_from_node(self, G, initial_state, connecting_node, edge_type):
        lace, _, _ = self.create_lace_from_initial_state(initial_state)
        G, start_node = connect_lace_to_graph(G, lace, connecting_node, edge_type=edge_type)
        # here you will have try and merge path that starts with X to the rest of the paths....
        G = try_merge_dynamics_to_graph(G, start_node)
        return G

    def get_graph_from_dynamics(self):
        pass

    def plot_graph(self, G):
        pass

    def process_graph(self, G=None):
        if G is None:
            G = self.load_file('graph')
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.set_node_attributes(G, 0, 'skip')
        G = make_compact(G)
        self.plot_graph(G)
        self.save_plot(plt, 'graph')
        self.save_file(G, 'compact_graph')
        return G

    def run(self):
        # from sklearn.metrics import mean_squared_error
        # score = mean_squared_error(self.outputs, self.Y.squeeze())
        # return score
        G = self.get_graph_from_dynamics();
        G = self.process_graph()
        return G


class DynamicsToGraphIntervalReproduction(DynamicsToGraphAbstract):
    name = 'dynamics_to_graph_interval_reproduction'
    value_descriptor = 'none'

    @property
    def dynamics_steps_delta(self):
        return 100

    @property
    def dynamics_steps_threshold(self):
        return 500

    def get_initial_set_states(self, initial_states):
        input = np.zeros((len(initial_states), 10, 2))
        input[:, :self.data_params.pulse, 1] = 1
        return self.model.run_system_from_inits(initial_states, steps=10, input_value=input)['state'][:, -1]

    def get_ready_initial_state(self):
        o_dict, s_dict = self.data_params.partition_to_epochs(self.outputs, self.states)
        initial_state = s_dict[50]['READY_SET'][15]
        return initial_state

    def get_pre_output_index(self, outputs):
        out = np.abs(outputs)
        q25, q75 = np.percentile(out, 10), np.percentile(out, 80)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        thresh = upper
        indices = np.argwhere(out < thresh).squeeze()
        return indices[-1]

    def plot_graph(self, G):
        plot_interval_reproduction_graph(G)

    def get_graph_from_dynamics(self):
        initial_state = self.get_ready_initial_state()
        G, dynamics, outputs = self.create_lace_from_initial_state(initial_state)
        initial_set_states = self.get_initial_set_states(dynamics)
        nodes = [20, self.get_pre_output_index(outputs)]

        for node in nodes:
            G = self.extend_graph_from_node(G, initial_set_states[node], node, edge_type=2)

        G = merge_components(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = try_connect_components(G, self.model)
        mark_output_nodes_rsg(G)
        self.save_file(G, 'graph')
        return G


class DynamicsToGraphIntervalDiscrimination(DynamicsToGraphAbstract):
    name = 'dynamics_to_graph_interval_discrimination'
    value_descriptor = 'none'

    @property
    def dynamics_steps_threshold(self):
        return 500

    @property
    def dynamics_steps_delta(self):
        return 100

    def get_initial_pulse1_states(self, initial_states, steps=10):
        set_input = self.generate_input1(len(initial_states), steps)
        return self.model.run_system_from_inits(initial_states, steps=steps, input_value=set_input)['state'][:, -1]

    def get_initial_pulse2_states(self, initial_states, steps=10):
        set_input = self.generate_input2(len(initial_states), steps)
        return self.model.run_system_from_inits(initial_states, steps=steps, input_value=set_input)['state'][:, -1]

    def get_initial_state(self, start_offset):
        inn = np.zeros((1, start_offset, 2))
        initial_state = self.model.predict(inn)['state'][0, -1]
        return initial_state

    def plot_graph(self, G):
        plot_interval_discrimination_graph(G)

    def get_graph_from_dynamics(self):
        start_offset = 10
        relax = 3
        initial_state = self.get_initial_state(start_offset)
        G, _, out = self.create_lace_from_initial_state(initial_state)

        # locations for 1st pulse
        t1s = [20 - start_offset]
        if np.argwhere(out).size == 0 or not is_there_a_cycle(G):
            t1s.append(len(G) - 1)
        for t1 in t1s:
            initial_e1_state = self.get_initial_pulse1_states(G.nodes[t1]['state'][None,:], relax)[0]
            G = self.extend_graph_from_node(G, initial_e1_state, t1, edge_type=4)
            lace_start = get_target_of_type(G, t1, 4)
            lace_path = get_path_from_source(get_zero_graph(G), lace_start)
            lace_dynamics = get_property_from_path(G, lace_path, 'state')
            #location for 2nd pulse
            t2s = sorted([15-relax, 25-relax, len(lace_dynamics)-1])
            if t1 > 30:
                t2s = t2s[::2]
            t2s = [c for c in t2s if c < len(lace_dynamics) and get_target_of_type(G, lace_path[c], 2) is None]

            initial_epoch2_states = self.get_initial_pulse2_states(lace_dynamics)
            # continue
            for t2 in t2s:
                G = self.extend_graph_from_node(G, initial_epoch2_states[t2], lace_path[t2], 2)

        G = merge_components(G)
        G = try_connect_components(G, model=self.model)
        mark_output_nodes(G)
        self.save_file(G, 'graph')
        return G


class DynamicsToGraphDelayedDiscrimination(DynamicsToGraphAbstract):
    name = 'dynamics_to_graph_delayed_discrimination'
    value_descriptor = 'none'

    def plot_graph(self, G):
        plot_delayed_discrimination_graph(G)

    def get_graph_from_dynamics(self):
        o_dict, s_dict = self.data_params.partition_to_epochs(self.outputs, self.states)
        # pred = self.model.predict(np.zeros((1, 200, 2)))
        G = create_lace_from_dynamics(s_dict[2][3]['START'], o_dict[2][3]['START'], try_locate_cycle=False)
        node_before_s1 = 4
        for t1 in [4, 8]:
            k = 9 if t1 == 10 else 10
            lace = create_lace_from_dynamics(s_dict[t1][k]['WAIT'][5:], o_dict[t1][k]['WAIT'][5:],
                                             try_locate_cycle=True)
            G, start_node = connect_lace_to_graph(G, lace, node_before_s1, t1)
            G = try_merge_dynamics_to_graph(G, start_node)
            sink = get_path_from_source(G, get_target_of_type(G, node_before_s1, t1))[-1]
            for t2 in [t1 - 2, t1 + 2]:
                lace = create_lace_from_dynamics(s_dict[t1][t2]['OUTPUT'][5:], o_dict[t1][t2]['OUTPUT'][5:],
                                                 try_locate_cycle=True)
                G, start_node = connect_lace_to_graph(G, lace, sink, t2)
                # here you will have try and merge path that starts with X to the rest of the paths....
                G = try_merge_dynamics_to_graph(G, start_node)


        mark_output_nodes(G, digit=0)
        self.save_file(G, 'graph')
        return G
