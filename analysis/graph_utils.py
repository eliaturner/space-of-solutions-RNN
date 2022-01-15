import numpy as np
import seaborn as sns

sns.set()
import networkx as nx


def is_there_a_cycle(G, source=None):
    try:
        nx.find_cycle(G, source)
        return True
    except:
        return False

def shift_nodes_names(G, offset):
    if offset:
        mapping = {i:i+offset for i in range(len(G))}
        G = nx.relabel_nodes(G, mapping)
    return G

def get_sink_from_node(G, node):
    while list(G.successors(node)):
        node = next(G.successors(node))

def get_nodes_with_attribute(G, attribute, value):
    return set([x for x,y in G.nodes(data=True) if y[attribute] == value])

def get_edges_with_attribute(G, attribute, value):
    return set([(s, t) for s, t, y in G.edges(data=True) if y[attribute] == value])


def remove_edge_by_attribute(G, s, t, attribute, value):
    for k in G[s][t].keys():
        if G[s][t][k][attribute] == value:
            G[s][t].pop(k)
            return

def get_property_from_path(G, path, property):
    state_list = [G.nodes[n][property] for n in path]
    return np.vstack(state_list)

def get_state_output_from_path(G, path):
    state = get_property_from_path(G, path, 'state')
    output = get_property_from_path(G, path, 'output')
    return state, output

def set_node_attributes(G, nodes, values, attribute):
    attr = {nodes[k]: values[k].squeeze() for k in range(min(len(nodes), len(values)))}
    G.add_nodes_from(list(attr.keys()))
    nx.set_node_attributes(G, attr, attribute)

def edges_from_path(path):
    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    return edges

def is_source(G, node, edge_type=0):
    if G.in_degree(node) > 0:
        for s in G.predecessors(node):
            if G[s][node]['edge_type'] == edge_type:
                return False

    return True


def get_edges_by_type(G, edge_type):
    return [(s, t) for (s, t) in G.edges() if G[s][t]['edge_type'] == edge_type]

def get_available_node(G):
    return max(list(G.nodes)) + 1

def mark_output_nodes(G, digit=0):
    for node in G.nodes:
        out = float(G.nodes[node]['output'])
        out = round(out, digit)
        G.nodes[node]['output'] = np.sign(out)

def mark_output_nodes_rsg(G):
    for node in G.nodes:
        out = float(G.nodes[node]['output'])
        out = round(out, 1)
        G.nodes[node]['output'] = 1 if  G.nodes[node]['output'] > 0.15 else 0
        # G.nodes[node]['output'] = max(np.sign(out), 0)

def get_edges_by_type(G, edge_type):
    return [(s, t) for (s, t) in G.edges() if G[s][t]['edge_type'] == edge_type]

def get_zero_graph(G):
    return get_subgraph(G, 0)

def get_target(G, s, edge_type):
    for t in G[s].keys():
        if G[s][t]['edge_type'] == edge_type:
            return t

    return None


def get_subgraph(G, edge_type):
    G_zero = G.copy()
    edge_types = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    edge_types.remove(edge_type)
    for e in edge_types:
        G_zero.remove_edges_from(get_edges_by_type(G, e))

    return G_zero


def has_source_of_type(G, t, edge_type):
    return bool(get_sources_of_type(G, t, edge_type))

def get_sources_of_type(G, t, edge_type):
    sources = G.predecessors(t)
    sources = [s for s in sources if G[s][t]['edge_type'] == edge_type]
    return sources

def get_target_of_type(G, s, edge_type):
    targets = G.successors(s)
    targets = [t for t in targets if G[s][t]['edge_type'] == edge_type] + [None]
    return targets[0]


def create_lace(s, o, offset=0):
    G = nx.path_graph(len(s), nx.DiGraph)
    nx.set_edge_attributes(G, 0, 'edge_type')
    node_to_state = {k: v for k, v in enumerate(s)}
    nx.set_node_attributes(G, node_to_state, 'state')
    node_to_output = {k: v for k, v in enumerate(o)}
    nx.set_node_attributes(G, node_to_output, 'output')
    G = shift_nodes_names(G, offset)
    return G


def find_simple_cycle(G):
    try:
        cycle = nx.find_cycle(G)
        if len(cycle) > 1:
            return [s for (s, _) in cycle]
        return []

    except nx.NetworkXNoCycle:
        return []

def is_zero_sink(G, node):
    if G.out_degree(node) > 0:
        for t in G.successors(node):
            if G[node][t]['edge_type'] == 0:
                return False

    return True