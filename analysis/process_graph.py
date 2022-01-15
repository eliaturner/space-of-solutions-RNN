import matplotlib.pyplot as plt
VERBOSE = False
from itertools import combinations, groupby
from matplotlib import cm
from collections import defaultdict
from analysis.graph_utils import *
'''
Auxilairy Functions
'''


def check_condition_nodes(G, n1, n2):
    if G.nodes[n1]['output'] != G.nodes[n2]['output']:
        return False

    parents = set(G.predecessors(n1)).intersection(set(G.predecessors(n2))).difference({n1, n2})
    for p in parents:
        if G[p][n1]['edge_type'] != G[p][n2]['edge_type'] and min(G[p][n1]['edge_type'], G[p][n2]['edge_type']) > 0:
            return False
    children = set(G.successors(n1)).intersection(set(G.successors(n2))).difference({n1, n2})
    for p in children:
        if G[n1][p]['edge_type'] != G[n2][p]['edge_type']:
            return False

    for edge_type in [2, 4]:
        s_target = get_target_of_type(G, n1, edge_type)
        t_target = get_target_of_type(G, n2, edge_type)
        targets = {s_target, t_target}.difference({None, n1, n2})
        if len(targets) == 2:
            return False
        targets2 = {s_target, t_target}.difference({None})
        if len(targets) == 1 and len(targets2) > 1:
            return False
        # if s_target and t_target and G.nodes[s_target]['output'] != G.nodes[t_target]['output']:
        #     return False

    return True


def check_condition_parallel_nodes(G, n1, n2):
    if get_target_of_type(G, n1, 0) != get_target_of_type(G, n2, 0):
        return False

    # if n1 in G[n2] or n2 in G[n1]:
    #     return False

    return check_condition_nodes(G, n1, n2)


def check_condition_zero_edge(G, s, t):
    if s == t or G[s][t]['edge_type'] > 0:
        return False

    return check_condition_nodes(G, s, t)


def check_square_condition(G, s, t):
    if s == t or G[s][t]['edge_type'] > 0:
        return False

    return check_square_condition1(G, s, t) or check_square_condition2(G, s, t)


def check_square_condition1(G, s, t):
    if get_target_of_type(G, s, 2) != get_target_of_type(G, t, 2):
        return False

    ready_s = get_target_of_type(G, s, 4)
    ready_t = get_target_of_type(G, t, 4)
    if ready_s is None or ready_t is None or ready_s == ready_t:
        return False
    ready_zero_s = get_target_of_type(G, ready_s, 0)
    ready_zero_t = get_target_of_type(G, ready_t, 0)
    if len({s, t, ready_zero_s, ready_zero_t}) > 2:
        return False

    ready_set_s = get_target_of_type(G, ready_s, 2)
    ready_set_t = get_target_of_type(G, ready_t, 2)
    if ready_set_s == ready_set_t:
        print('Voila')
        return True

    return False


def check_square_condition2(G, s, t):
    # return True
    if get_target_of_type(G, s, 4) != get_target_of_type(G, t, 4):
        return False

    ready_s = get_target_of_type(G, s, 2)
    ready_t = get_target_of_type(G, t, 2)
    if ready_s is None or ready_t is None or ready_s == ready_t:
        return False

    if G.nodes[ready_s]['output'] != G.nodes[ready_t]['output'] or G.nodes[ready_t]['output'] == 0:
        return False

    ready_zero_s = get_target_of_type(G, ready_s, 0)
    ready_zero_t = get_target_of_type(G, ready_t, 0)
    if len({s, t, ready_zero_s, ready_zero_t}) > 2:
        return False

    return True


'''
End
'''

def edge_to_node(G, s, t):
    G.nodes[s]['skip'] += G.nodes[t]['skip']
    G.remove_edge(s, t)
    return contract_nodes(G, s, t)

def contract_nodes(G, s1, s2, skips=None):
    if s2 == 0:
        s1, s2 = s2, s1

    common_parents = set(G.predecessors(s1)).intersection(set(G.predecessors(s2)))
    for p in common_parents:
        nonzero_t = s1 if G[p][s1]['edge_type'] > 0 else s2
        G.add_edge(p, p, edge_type=G[p][nonzero_t]['edge_type'])
        G.remove_edge(p, nonzero_t)

    G = nx.contracted_nodes(G, s1, s2, self_loops=True)
    return G


def check_condition_mixed_edge(G, s, t):
    if not(t in G[s] and G[s][t]['edge_type'] == 0 and G.nodes[s]['output'] == 0 and G.nodes[t]['output'] != 0):
        return False
    if G.out_degree(s) == 1 and G.in_degree(t) == 1 and get_target_of_type(G, t, 0) != s:
        return True


def redirect_output_routes(G):
    G_set = get_subgraph(G, 2)
    nodes = [node for node in G.nodes if G.in_degree(node) == G_set.in_degree(node) and G.in_degree(node) > 0]

    for node in nodes:
        target = get_target_of_type(G, node, 0)
        if target is None or G.nodes[target]['output'] != 0:
            continue
        visited = {target}
        new_target = get_target(G, target, 0)
        while new_target is not None and new_target not in visited and G.nodes[new_target]['output'] == 0:
            G.remove_edge(node, target)
            target = new_target
            G.add_edge(node, target, edge_type=0)
            visited.add(target)
            new_target = get_target(G, target, 0)

    return G


def merge_zero_one_edges(G):
    if VERBOSE:
        print(f'removing mixed edges, size:{len(G)}')

    while True:
        zero_condition = (e for e in G.edges() if check_condition_mixed_edge(G, e[0], e[1]))
        e = next(zero_condition, None);
        if e:
            G = nx.contracted_nodes(G, e[1], e[0], self_loops=False)
        else:
            return G


def merge_square(G):
    if VERBOSE:
        print(f'removing square edges, size:{len(G)}')

    while True:
        square_condition = (e for e in G.edges() if check_square_condition(G, e[0], e[1]))
        e = next(square_condition, None);
        if e:
            # G.remove_edge(e[0], e[1])
            for edge_type in [2, 4]:
                ready_s = get_target_of_type(G, e[0], edge_type)
                ready_t = get_target_of_type(G, e[1], edge_type)
                if len({ready_s, ready_t}.difference({None})) == 2:
                    G = nx.contracted_nodes(G, ready_s, ready_t, self_loops=False)
            G = nx.contracted_nodes(G, e[0], e[1], self_loops=False)
        else:
            return G


def merge_edges(G):
    if VERBOSE:
        print(f'removing zero edges, size:{len(G)}')

    while True:
        zero_condition = (e for e in G.edges() if check_condition_zero_edge(G, e[0], e[1]))
        e = next(zero_condition, None);
        if e:
            G = edge_to_node(G, e[0], e[1])
        else:
            return G


def merge_lateral_nodes(G):
    if VERBOSE:
        print(f'removing parallel edges, size:{len(G)}')

    while True:
        # cycle = find_simple_cycle(get_zero_graph(G))
        nodes = (node for node in G.nodes() if G.in_degree(node) > 1)
        for node in nodes:
            x = lambda pair: check_condition_parallel_nodes(G, pair[0], pair[1])
            parents = sorted(G.predecessors(node))
            parents = [p for p in parents if G[p][node]['edge_type'] == 0]
            pair = next(filter(x, combinations(parents, r=2)), None)
            if pair:
                G.nodes[pair[0]]['skip'] += G.nodes[pair[1]]['skip']
                G = contract_nodes(G, pair[0], pair[1])
                break
            else:
                continue
        else:
            return G


def get_connected_components(G):
    connected_components = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
    return connected_components

def make_compact(G):
    processes = [merge_edges, merge_lateral_nodes, merge_zero_one_edges, redirect_output_routes, merge_square]
    while True:
        n_nodes, n_edges = len(G), len(G.edges)
        G = remove_uninformative_paths(G, 0)
        for func in processes:
            G = func(G)
        # plot_graph(copy_g); plt.show()
        if len(G) == n_nodes and n_edges == len(G.edges):
            break

    return G


def merge_path_to_edge(G, path, edge_type):
    s = get_sources_of_type(G, path[0], edge_type)[0]
    t = get_target_of_type(G, path[-1], edge_type)
    num_skips = sum(G.nodes[n]['skip'] for n in path)
    G.nodes[s]['skip'] += num_skips
    G.remove_nodes_from(path)
    G.add_edge(s, t, edge_type=edge_type)
    return G

def get_path_from_source(G, source, component=None):
    if component is None:
        component = nx.descendants(G, source)
        component.add(source)
    try:
        cycle_edges = nx.find_cycle(G, source)
        path = [e[0] for e in cycle_edges]
        path = nx.shortest_path(G, source, path[-1])
    except:
        path = nx.dag_longest_path(nx.subgraph(G, component), source)

    return path

def get_paths(G):
    paths = defaultdict(list)
    visited_nodes = set()
    for idx, component in enumerate(nx.weakly_connected_components(G)):
        try:
            cycle = [e[0] for e in nx.find_cycle(nx.subgraph(G, component))]
            if len(cycle) > 1:
                visited_nodes.update(cycle)
                paths[idx].append(cycle)
        except nx.NetworkXNoCycle:
            pass
        sources = [node for node in component if G.in_degree(node) == 0]
        for source in sources:
            path = get_path_from_source(G, source)
            for i in range(len(path)):
                if path[i] in visited_nodes:
                    path = path[:i+1]
                    break

            if len(path) > 1:
                paths[idx].append(path)
            visited_nodes.update(path)

        paths[idx] = paths[idx][-1:] + paths[idx][:-1]

    return paths

def get_zero_paths(G, edge_type = 0, force_nodes = False):
    G = get_zero_graph(G)
    non_zero_nodes = [node for node in G.nodes if G.nodes[node]['output']]
    if force_nodes:
        G.remove_nodes_from(non_zero_nodes)

    return get_paths(G)


def get_longest_path(G):
    try:
        cycle = nx.find_cycle(G)

        long_path = [s for (s, _) in cycle]
        roots = (v for v, d in G.in_degree() if d == 0)
        target = long_path[0]
        paths = [list(nx.all_simple_paths(G, root, target)) for root in roots]
        paths = [path[0] for path in paths if len(path)]
        paths.append(long_path)
        lens = [len(path) for path in paths]
        max_len = max(lens)
        for path in paths:
            if len(path) == max_len:
                return path

        print()
    except nx.NetworkXNoCycle:
        long_path = nx.dag_longest_path(G)

    return long_path


def is_node_uninformative(G, node):
    if node == 0 or G.in_degree(node) > 1 or G.out_degree(node) > 1 or (get_target(G, node, 0) is None):
        return False

    return True


def remove_uninformative_paths(G, node_type):
    G_zero = get_zero_graph(G)
    nodes_to_remove = [node for node in G_zero.nodes if G.nodes[node]['output'] != node_type]
    G_zero.remove_nodes_from(nodes_to_remove)
    while True:
        paths = [path[1:-1] for paths in get_paths(G_zero).values() for path in paths]
        func = lambda x: is_node_uninformative(G, x)
        paths = [[e for e in g] for path in paths for k, g in groupby(path, func) if k]
        paths = [path for path in paths if len(path) > 0]
        if len(paths) == 0:
            # return G
            while True:
                edges = (e for e in G_zero.edges if G.in_degree(e[0]) == 1 and G.in_degree(e[1]) == 1 and G.out_degree(e[0]) == 1 and e[1] != 0)
                e = next(edges, None)
                if e:
                    G.nodes[e[0]]['skip'] += G.nodes[e[1]]['skip']
                    G = nx.contracted_nodes(G, e[0], e[1], self_loops=False)
                    G_zero = nx.contracted_nodes(G_zero, e[0], e[1], self_loops=False)
                else:
                    return G

        for path in paths:
            G = merge_path_to_edge(G, path, 0)
            G_zero = merge_path_to_edge(G_zero, path, 0)


def plot_delayed_discrimination_graph(G):
    dd = {}
    for node in G.nodes:
        if node == 0:
            dd[node] = 0
        elif G.nodes[node]['output'] != 0:
            dd[node] = 2
        else:
            dd[node] = 1

    nx.set_node_attributes(G, dd, 'subset')
    pos = nx.multipartite_layout(G)
    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())
    zero_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 0]
    plt.clf()
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=edges, edge_color=edges_color, edge_cmap=cm.Reds, edge_vmin=0, edge_vmax=max(edges_color) + 1, width=2, arrowsize=10);
    nx.draw_networkx_nodes(G, pos, nodes, node_color=nodes_color, cmap='PiYG', vmin=-1, vmax=1, edgecolors='black', node_size=300);
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.2', edgelist=zero_edges, edge_color='black', width=2, arrowsize=10);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=300);


def plot_interval_discrimination_graph(G, bipartite=True):
    G = G.copy()
    plt.clf();
    if len(G) == 2:
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = nx.circular_layout(G)
    if bipartite:
        if len(G) == 2:
            pos = nx.bipartite_layout(G, G.nodes())
        else:
            zero_path = get_path_from_source(get_zero_graph(G), 0)
            dd = {}
            for node in G.nodes:
                if node in zero_path:
                    dd[node] = 0
                elif G.nodes[node]['output'] != 0:
                    dd[node] = 2
                else:
                    dd[node] = 1

            nx.set_node_attributes(G, dd, 'subset')
            pos = nx.multipartite_layout(G)
            sorted_positions = sorted([pos[node] for node in zero_path], key= lambda x: x[1])
            for i, node in enumerate(zero_path):
                pos[node] = sorted_positions[i]

    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())

    set_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 2]
    ready_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 4]
    zero_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 0]

    set_edges, ready_edges = ready_edges, set_edges

    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=set_edges, edge_color='gold', width=2, arrowsize=10);
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=ready_edges, edge_color='orange', width=2, arrowsize=10);
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=zero_edges, edge_color='black', width=2, arrowsize=10);
    nx.draw_networkx_nodes(G, pos, nodes, node_color=nodes_color, cmap='PiYG', vmin=-1, vmax=1, edgecolors='black', node_size=300);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=300);


def plot_interval_reproduction_graph(G, bipartite=True):
    G = G.copy()
    plt.clf();
    if len(G) == 2:
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = nx.circular_layout(G)
    if bipartite:
        if len(G) == 2:
            pos = nx.bipartite_layout(G, G.nodes())
        else:
            zero_path = [0] + get_path_from_source(get_zero_graph(G), 0)
            pos = nx.bipartite_layout(G, zero_path)
            sorted_positions = sorted([pos[node] for node in zero_path], key= lambda x: x[1])
            for i, node in enumerate(zero_path):
                pos[node] = sorted_positions[i]

            not_zero_path = sorted([node for node in G.nodes if node not in zero_path])
            sorted_positions = sorted([pos[node] for node in not_zero_path], key= lambda x: x[1])
            for i, node in enumerate(not_zero_path):
                pos[node] = sorted_positions[i]

    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=edges, edge_color=edges_color, edge_cmap=cm.hot, edge_vmin=0, edge_vmax=max(edges_color) + 1, width=2, arrowsize=10);
    nx.draw_networkx_nodes(G, pos, nodes, node_color=nodes_color, cmap='PiYG', vmin=-1, vmax=1, edgecolors='black', node_size=300);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=300);



