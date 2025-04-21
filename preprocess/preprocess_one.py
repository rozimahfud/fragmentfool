import os

import networkx as nx

from glob import glob
from tqdm import tqdm

from utils.graphutils import *
from preprocess.joern_preprocess import *

def preprocess_one(path,targets):
    datalist = []

    for p in tqdm(glob(path+"*")):
        file_name = int(p.split("/")[-1].split(".")[0])
        y = targets[file_name]

        cpg = nx.Graph(nx.nx_pydot.read_dot(p))

        file_path = "data/devign/c-files/" + str(file_name) + ".c"
        out_path = "data/devign/ast/"
        dot_file_path = out_path + str(file_name) + "/1-ast.dot"

        if not os.path.exists(dot_file_path):
            result = generate_JOERN(file_path,out_path,"ast")

        ast = nx.Graph(nx.nx_pydot.read_dot(dot_file_path))
        ast = cpg_to_ast(cpg,ast)

        new_ast = nx.dfs_tree(ast, source=list(ast.nodes)[0])
        new_ast.add_nodes_from((i, ast.nodes[i]) for i in new_ast.nodes)

        data = numpy_to_pyg(new_ast,y)
        datalist.append(data)

    return datalist