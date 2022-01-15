from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import models.model_config as model_config
from analysis.dynamics_to_graph import DynamicsToGraphIntervalReproduction, DynamicsToGraphIntervalDiscrimination, DynamicsToGraphDelayedDiscrimination
from analysis.process_graph import plot_interval_reproduction_graph, plot_delayed_discrimination_graph, plot_interval_discrimination_graph
from tools.utils import dump_pickle, load_pickle


def plot_cluster_histogram(sizes, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    # fig.subplots_adjust(top=0.85)
    sizes = sorted(sizes)[::-1]
    limit = np.argwhere(np.array(sizes) < 10).squeeze()[0]
    num_taken = sum(sizes[:limit])
    percentage = int(100*num_taken/400)
    labels = len(sizes)*['']
    # sizes.append(400 - sum(sizes))

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


def find_isomorphic_classes(graph_list):
    isomorphic_classes = []
    class_to_list = defaultdict(list)
    classes = []
    for idx1, G1 in enumerate(graph_list):
        for idx2, G2 in enumerate(isomorphic_classes):
            if nx.is_isomorphic(G1, G2):
                if 0 in G1[0] and 0 in G2[0] and G1[0][0]['edge_type'] !=  G2[0][0]['edge_type']:
                    continue
                s1 = set([G1[0][t]['edge_type'] for t in G1.successors(0)])
                s2 = set([G2[0][t]['edge_type'] for t in G2.successors(0)])
                edge_values1 = set(nx.get_edge_attributes(G1, 'edge_type').values())
                edge_values2 = set(nx.get_edge_attributes(G2, 'edge_type').values())
                if edge_values1 == edge_values2 and (s1 == s2 or 2 not in edge_values2):
                    class_to_list[idx2].append(idx1)
                    classes.append(idx2)
                    break
        else:
            class_to_list[len(isomorphic_classes)].append(idx1)
            classes.append(len(isomorphic_classes))
            isomorphic_classes.append(G1)

    return class_to_list, isomorphic_classes


def graphs_to_labels(class_to_list, size):
    y = np.zeros(size)
    tops = [graph for graph in class_to_list.keys() if len(class_to_list[graph]) >= 10]
    for label, graph in enumerate(tops):
        for net in class_to_list[graph]:
            y[net] = label + 1

    return y

def return_dict_sorted_by_size(d):
    l = [(key, len(d[key])) for key in d.keys()]
    l = sorted(l, key=lambda p : p[1])[::-1]
    l = [a[0] for a in l]
    d_new = {}
    for i, graph in enumerate(l):
        d_new[i] = graph

    return d_new

def plot_main_clusters(class_to_list, isomorphic_classes, name, plotter):
    mapp = return_dict_sorted_by_size(class_to_list)
    class_to_list = {i:class_to_list[mapp[i]] for i in mapp.keys()}
    isomorphic_classes = {i:isomorphic_classes[mapp[i]] for i in mapp.keys()}
    for graph, l in class_to_list.items():
        if len(l) < 10:
            break
        plotter(isomorphic_classes[graph])
        plt.savefig(f'figures/{name}_{graph}.pdf')

    return class_to_list

def graphs_to_clusters(graphs, task, architecture, plot_func, cluster_pairs):
    class_to_list, isomorphic_classes = find_isomorphic_classes(graphs)
    for p1, p2 in cluster_pairs:
        class_to_list[p2].extend(class_to_list[p1])
        class_to_list.pop(p1)
    sizes = [len(class_to_list[graph]) for graph in class_to_list.keys()]
    class_to_list = plot_main_clusters(class_to_list, isomorphic_classes, f'{task}_{architecture}', plot_func)
    return class_to_list

def rsg_gru_stuff():
    graphs = [load_pickle(f'models/rsg_2c_50_gru_{n}/i{instance}/compact_graph.pkl') for n in range(20, 60, 10)
                  for instance in range(100, 200)]
    cluster_pairs = [(5, 2), (3, 4), (11, 2), (6, 0), (13, 1), (16, 2), (19, 2), (27, 14), (31, 2)]
    return graphs_to_clusters(graphs, 'rsg', 'gru', plot_interval_reproduction_graph, cluster_pairs)


def rsg_lstm_stuff():
    cluster_pairs = [(9, 4), (10, 5)]
    graphs = [load_pickle(f'models/rsg_2c_50_lstm_{n}/i{instance}/compact_graph.pkl') for n in range(20, 60, 10)
                  for instance in range(100, 200)]

    return graphs_to_clusters(graphs, 'rsg', 'lstm', plot_interval_reproduction_graph, cluster_pairs)

def all_stuff():
    graphs = [load_pickle(f'models/rsg_2c_50_{name}_{n}/i{instance}/compact_graph.pkl') for name in ['vanilla', 'gru', 'lstm'] for n in range(20, 60, 10)
                  for instance in range(100, 200)]
    cluster_pairs = [(9, 1), (13,0), (11, 3), (12,10)]
    class_to_list = graphs_to_clusters(graphs, 'rsg', 'all', plot_interval_reproduction_graph, cluster_pairs)
    y = np.zeros(len(graphs))
    for k in class_to_list.keys():
        for v in class_to_list[k]:
            y[v] = k

    y = y.reshape((3, 400))
    lim = 6
    hist_vanilla = np.bincount(np.array(y[0], dtype=int))[:lim]
    hist_gru = np.bincount(np.array(y[1], dtype=int))[:lim]
    hist_lstm = np.bincount(np.array(y[2], dtype=int))[:lim]
    fig, ax = plt.subplots()
    ax.bar(np.arange(lim), hist_vanilla, color='blue', label='Vanilla')
    ax.bar(np.arange(lim), hist_gru, bottom=hist_vanilla, color='orange', label='GRU')
    ax.bar(np.arange(lim), hist_lstm, bottom=hist_gru + hist_vanilla, color='green', label='LSTM')
    ax.set_xticks([])
    plt.grid(False)
    ax.set_facecolor('white')
    plt.legend()
    plt.savefig(f'figures/RSG_ALL_MATRIX_TOPOLOGIES/hist_architecture.pdf', bbox_inches='tight')

def delayed_discrimination_vanilla():
    graphs = [graph for wrapper in model_config.delayed_discrimination_vanilla_models() for graph in wrapper.get_analysis([DynamicsToGraphDelayedDiscrimination])]
    return graphs_to_clusters(graphs, 'delayed_discrimination', 'vanilla', plot_delayed_discrimination_graph, [])

def interval_discrimination_vanilla():
    graphs = [graph for wrapper in model_config.interval_discrimination_vanilla_models() for graph in wrapper.get_analysis([DynamicsToGraphIntervalDiscrimination])]
    return graphs_to_clusters(graphs, 'interval_discrimination', 'vanilla', plot_interval_discrimination_graph, [])

def interval_reproduction_vanilla():
    graphs = [graph for wrapper in model_config.interval_reproduction_vanilla_models() for graph in wrapper.get_analysis([DynamicsToGraphIntervalReproduction])]
    return graphs_to_clusters(graphs, 'interval_reproduction', 'vanilla', plot_interval_reproduction_graph, [])

def interval_reproduction_gru():
    graphs = [graph for wrapper in model_config.interval_reproduction_gru_models() for graph in wrapper.get_analysis([DynamicsToGraphIntervalReproduction])]
    return graphs_to_clusters(graphs, 'interval_reproduction', 'gru', plot_interval_reproduction_graph, [])

def interval_reproduction_lstm():
    graphs = [graph for wrapper in model_config.interval_reproduction_lstm_models() for graph in wrapper.get_analysis([DynamicsToGraphIntervalReproduction])]
    return graphs_to_clusters(graphs, 'interval_reproduction', 'lstm', plot_interval_reproduction_graph, [])

if __name__ == '__main__':
    interval_reproduction_gru()
    interval_reproduction_lstm()


