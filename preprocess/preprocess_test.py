import torch
from utils.graphutils import *

def preprocess_one(graph,y,node_type_dict):
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