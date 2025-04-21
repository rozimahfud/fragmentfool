import os
import sys
import random
import numpy as np
import networkx as nx

from glob import glob
from tqdm import tqdm

from perturbation.perturb_one import *
from utils.graphutils import *
from preprocess.joern_preprocess import *

def select_perturbation(perturb_type,path,targets,k,N,perturb_target,node_type_dict):
    datalist = []
    perturb_list = generate_binary_list(N,k)

    for p in tqdm(glob(path+"*")):
        file_name = int(p.split("/")[-1].split(".")[0])
        y = targets[file_name]

        cpg = nx.Graph(nx.nx_pydot.read_dot(p))

        file_path = "data/devign/c-files/"+str(file_name)+".c"
        out_path = "data/devign/ast/"
        dot_file_path = out_path + str(file_name) + "/1-ast.dot"

        if not os.path.exists(dot_file_path):
            result = generate_JOERN(file_path,out_path,"ast")

        ast = nx.Graph(nx.nx_pydot.read_dot(dot_file_path))

        if y == perturb_target:
            isPerturb = perturb_list.pop()
            
            if isPerturb == 1:
                if perturb_type == "1":
                    new_ast = perturb_one(cpg,ast,file_name)
                else:
                    sys.exit("== There is no such model name ==")
            else:
                new_ast = create_new_ast_no_perturb(cpg,ast)
        else:
            new_ast = create_new_ast_no_perturb(cpg,ast)

        data = numpy_to_pyg(new_ast,y,node_type_dict)
        datalist.append(data)

    return datalist

def create_new_ast_no_perturb(cpg,ast):
    ast = cpg_to_ast(cpg, ast)
    new_ast = nx.dfs_tree(ast, source=list(ast.nodes)[0])
    new_ast.add_nodes_from((i, ast.nodes[i]) for i in new_ast.nodes)
    
    return new_ast

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
