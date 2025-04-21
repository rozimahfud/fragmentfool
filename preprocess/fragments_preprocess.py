import random
import networkx as nx

def find_random_tuple_same_degree(graph, tuple_size=2):
    # Count the number of neighbors (degree) for each node
    degree_dict = dict(graph.degree())
    
    # Group nodes by their degree
    degree_to_nodes = {}
    for node, degree in degree_dict.items():
        if degree not in degree_to_nodes:
            degree_to_nodes[degree] = []
        degree_to_nodes[degree].append(node)
    
    # Find degrees that have at least 'tuple_size' nodes
    valid_degrees = [degree for degree, nodes in degree_to_nodes.items() 
                    if len(nodes) >= tuple_size]
    
    if not valid_degrees:
        return None  # No valid tuples found
    
    # Pick a random degree
    chosen_degree = random.choice(valid_degrees)
    
    # Select random nodes with that degree
    chosen_nodes = random.sample(degree_to_nodes[chosen_degree], tuple_size)
    
    return chosen_nodes, chosen_degree

def make_fragments(graph):
    fragment_list = []
    
    root_nodes = []
    for e in graph.edges:
        if e[0] not in root_nodes:
            root_nodes.append(e[0])
    
    fragment_graph = nx.DiGraph()
    for i, root in enumerate(root_nodes):
        neigh = list(graph.neighbors(root))
        childs = neigh[1:]
        frags = make_subgraph(root,childs)
        fragment_list.append(frags)
        # draw_graph(frags,str(i)+"_frags")

        if not i == 0:
            parent = neigh[0]
            fragment_graph.add_edge(parent,root)

    return fragment_list, fragment_graph

def make_subgraph(root,childs):
    graph = nx.Graph()
    graph.add_node(root)

    for c in childs:
        graph.add_edge(root,c)

    return graph