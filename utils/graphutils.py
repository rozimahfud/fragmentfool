import torch
import random
import networkx as nx
from utils.datautils import *
from torch_geometric.data import Data

def create_list_graph_object(path,out_path,dataset_name, graph_repret):
    NODE_TYPE_DICT = set()
    if graph_repret == "cpg":
        for p in tqdm(glob(path+"/*")):
            file_name = p.split("/")[-1]
            dot_file_path = p + "/" + file_name +".c/_global_.dot"
            graph = nx.Graph(nx.nx_pydot.read_dot(dot_file_path))

            for node in graph.nodes:
                label = graph.nodes[node]["label"]

                if label:
                    graph.nodes[node]["label"] = label
                    NODE_TYPE_DICT.add(label)
                else:
                    graph.nodes[node]["label"] = "UNKNOWN"
                    NODE_TYPE_DICT.add("UNKNOWN")

            nx.nx_pydot.write_dot(graph,out_path + file_name + ".dot")

        save_as_pickle(NODE_TYPE_DICT,"data/"+dataset_name+"/node-type-dict.pickle")
        
    else:
        sys.exit("== There is no such graph representation format ==")

def cpg_to_ast(cpg,ast):
    # we use ast as a base and then find node label in cpg
    for n in ast.nodes:
        if n not in list(cpg.nodes):
            cpg_label = '"UNKNOWN"'
            cpg_code = ""
        else:
            cpg_label = cpg.nodes[n]["label"]
            cpg_code = cpg.nodes[n]["CODE"]

        ast.nodes[n]["label"] = cpg_label
        ast.nodes[n]["CODE"] = cpg_code

    mapping = dict(zip(list(ast.nodes),list(range(len(ast.nodes)))))
    ast = nx.relabel_nodes(ast, mapping)

    return ast

def swap_grand_childs(childs_a,childs_b,graph):
    
    for a,b in zip(childs_a,childs_b):
        grand_a = list(graph.neighbors(a))[1:]
        grand_b = list(graph.neighbors(b))[1:]
        
        # remove all edges of grand children for both
        for g_a in grand_a:
            graph.remove_edge(a,g_a)

        for g_b in grand_b:
            graph.remove_edge(b,g_b)

        # swapping
        for g_a in grand_a:
            graph.add_edge(b,g_a)

        for g_b in grand_b:
            graph.add_edge(a,g_b)

    return graph

def numpy_to_pyg(graph,y,node_type_dict):
    mapping = dict(zip(list(graph.nodes),list(range(len(graph.nodes)))))
    graph = nx.relabel_nodes(graph, mapping)

    # make node features:
    node_feat = []
    for node in graph.nodes:
        label = graph.nodes[node]["label"]
        node_feat.append(one_hot_embedding(label,node_type_dict))
    x = torch.tensor(node_feat)
    
    edge_index = []
    for edge in graph.edges():
        # Each edge is added in both directions for undirected graphs
        edge_index.append([edge[0], edge[1]])
        if not graph.is_directed():
            edge_index.append([edge[1], edge[0]])
    
    # Convert to tensor and transpose to get the PyG format [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)

def one_hot_embedding(node_type,node_type_dict):
    embed = [0] * len(node_type_dict)
    embed[node_type_dict[node_type]] = 1
    return embed

def generate_binary_list(N, percentage_of_ones):
    
    # Validate inputs
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    
    if not (0 <= percentage_of_ones <= 100):
        raise ValueError("percentage_of_ones must be between 0 and 100")
    
    # Calculate the number of ones
    num_ones = int(round(N * percentage_of_ones / 100))
    
    # Ensure number of ones doesn't exceed N
    num_ones = min(num_ones, N)
    
    # Create a list with the appropriate number of ones and zeros
    binary_list = [1] * num_ones + [0] * (N - num_ones)
    
    # Shuffle the list to randomize the positions of ones
    random.shuffle(binary_list)
    
    return binary_list