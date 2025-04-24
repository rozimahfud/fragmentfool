import networkx as nx
import pandas as pd
import numpy as np
import random
import pickle
import os
import sys

from tqdm import tqdm
from glob import glob

from preprocess.joern_preprocess import *

def main():
    # Hyper parameter and setup
    dataset_name = "big-vul"
    graph_repret = "cpg"
    model_name = "GatedGraphConv"
    saved_model_name = "model-test"
    perturb_target = 1
    pre_method = "1"
    perturb_type = "1"
    learning_rate = 1e-5
    weight_decay = 1.3e-3
    k = 50
    
    test_size = 0.2
    random_state = 42
    batch_size = 4
    num_epochs = 100
    start_model_number = 0
    cont = False
    split_set = ["test","train"]

    dataset_raw = "../dataset-preparations/"

    file_path = dataset_raw+dataset_name+"/"

    # Generate graph representation using JOERN. Ouutput as DOT files.
    folder_path = "../data/"+dataset_name+"/"+graph_repret+"/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("\n== Generating graph representation ==\n")
        for sp in split_set:
            for p in tqdm(glob(file_path+sp+"/*.c")):
                result = generate_JOERN(p,folder_path,graph_repret)
                if not result==0:
                    print("Path:",p,"is not success.")  
    else:
        if len(glob(file_path+"**/*.c",recursive=True)) != len(os.listdir(folder_path)):
            print("\n== Continuing generating graph representation ==\n")
        
            for sp in split_set:
                ref_data = pd.read_csv("../dataset-preparations/big-vul/big_"+sp+"_adj.csv")

                for p in tqdm(glob(file_path+"test/*.c")):
                    file_name = p.split("/")[-1].split(".")[0]
                    if int(file_name) in ref_data.index.tolist():
                        if os.path.exists(folder_path+file_name):
                            continue
                        else:
                            result = generate_JOERN(p,folder_path,graph_repret)
                            if not result==0:
                                print("Path:",p,"is not success.")

        print("\n== Graph representation has been generated in",folder_path,"==\n")


if __name__ == "__main__":
    main()