import networkx as nx
import pandas as pd
import numpy as np
import random
import pickle
import os
import sys
import torch

from tqdm import tqdm
from glob import glob

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import torch.nn as nn

from process.model import Net
from process.running import *
from utils.datautils import *
from utils.drawutils import *
from preprocess.joern_preprocess import *
from preprocess.fragments_preprocess import *
from preprocess.main_preprocess_test import *
from perturbation.select_perturbation import *

def model_setup(model_name, learning_rate, weight_decay,cont_path=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "GatedGraphConv":
        if not cont_path:
            model_config = {
                "gated_graph_conv_args": {"out_channels" : 200, "num_layers" : 4, "aggr" : "add", "bias": True},
                "conv_args": {
                    "conv1d_1" : {"in_channels": 228, "out_channels": 50, "kernel_size": 3, "padding" : 1},
                    "conv1d_2" : {"in_channels": 50, "out_channels": 20, "kernel_size": 1, "padding" : 1},
                    "maxpool1d_1" : {"kernel_size" : 3, "stride" : 2},
                    "maxpool1d_2" : {"kernel_size" : 2, "stride" : 2}
                    },
                "emb_size" : 101,
                "dropout_rate" : 0.5
                }
            model = Net(**model_config, device=device)
        else:
            model = torch.load(cont_path)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    return model, criterion, optimizer, device

def main():
    # Hyper parameter and setup
    dataset_name = "devign"
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

    dataset_raw = "../dataset-preparations/"
    model_path = "data/"+dataset_name+"/"+saved_model_name+"/perturbed/"+model_name+"/"
    cont_path = "data/"+dataset_name+"/"+saved_model_name+"/perturbed/"+model_name+"/"+"/model"+str(start_model_number)+".pth"
    naming_running = dataset_name + "_" + pre_method + "_" + model_name + "_" + perturb_type + "_" + str(perturb_target) + "_" + str(k)

    file_path = dataset_raw+dataset_name+"/"
    node_type_dict_path = "./data/"+dataset_name+"/node-type-dict.pickle"

    # # Preparing dataset based in dataset name
    # dataset = prepare_data(dataset_raw,dataset_name)

    # # Writing all functions in dataset into C files
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    #     print("\n== Making c files saved in",file_path,"==\n")
    #     make_file(dataset,file_path)
    # else:
    #     print("\n== c files are available in",file_path,"==\n")

    # Generate graph representation using JOERN. Ouutput as DOT files.
    folder_path = "../data/"+dataset_name+"/"+graph_repret+"/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("\n== Generating graph representation ==\n")
        for p in tqdm(glob(file_path+"**/*.c",recursive=True)):
            result = generate_JOERN(p,folder_path,graph_repret)
            if not result==0:
                print("Path:",p,"is not success.")  
    else:
        if len(glob(file_path+"**/*.c",recursive=True)) != len(os.listdir(folder_path)):
            print("\n== Continuing generating graph representation ==\n")
            for p in tqdm(glob(file_path+"**/*.c",recursive=True))::
                file_name = p.split("/")[-1].split(".")[0]
                if os.path.exists(folder_path+filename):
                    continue
                else:
                    result = generate_JOERN(p,folder_path,graph_repret)
                    if not result==0:
                        print("Path:",p,"is not success.")

        print("\n== Graph representation has been generated in",folder_path,"==\n")

    # # Creating graph object
    # out_path = "./data/"+dataset_name+"/graph-object-"+graph_repret+"/"
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    #     print("\n== Creating graph object ==\n")
    #     input_path = "./data/"+dataset_name+"/"+graph_repret
    #     create_list_graph_object(input_path,out_path,dataset_name,graph_repret)
    # else:
    #     print("\n== Graph object has been created in ",out_path,"==\n")

    # # Importing node type dictionary
    # node_type_dict_path = "data/"+dataset_name+"/node-type-dict.pickle"
    # if os.path.exists(node_type_dict_path):
    #     # load node type dictionary
    #     NODE_TYPE_DICT = read_pickle_file(node_type_dict_path)
    #     NODE_TYPE_DICT = dict(zip(NODE_TYPE_DICT,list(range(len(NODE_TYPE_DICT)))))
    #     print("\n== Loaded node type dictionary ==\n")
    #     print(NODE_TYPE_DICT)
    # else:
    #     sys.exit("\n== Node type dictionary is not found ==\n")

    # # Making dictionary for label
    # targets = dict()
    # for f in os.listdir(out_path):
    #     file_name = int(f.split("/")[-1].split(".")[0])
    #     targets[file_name] = dataset.iloc[int(file_name)]["target"]

    # # Splitting train test
    # train_test_path = "data/"+dataset_name+"/split-test-perturb-"+naming_running+"/"
    # if not os.path.exists(train_test_path):
    #     os.makedirs(train_test_path)

    #     data_train_idx, data_test_idx, y_train, y_test = train_test_split(list(targets.keys()),
    #                                                                   list(targets.values()),
    #                                                                   test_size=test_size,
    #                                                                   random_state=random_state)

    #     save_as_pickle(data_train_idx,train_test_path+"data_train_idx.pickle")
    #     save_as_pickle(data_test_idx,train_test_path+"data_test_idx.pickle")
    #     save_as_pickle(y_train,train_test_path+"y_train.pickle")
    #     save_as_pickle(y_test,train_test_path+"y_test.pickle")

    #     print("\n== Splitting dataset saved in",train_test_path,"==\n")
    # else:
    #     data_train_idx = read_pickle_file(train_test_path+"data_train_idx.pickle")
    #     data_test_idx = read_pickle_file(train_test_path+"data_test_idx.pickle")
    #     y_train = read_pickle_file(train_test_path+"y_train.pickle")
    #     y_test = read_pickle_file(train_test_path+"y_test.pickle")

    #     print("\n== Splitting dataset has finished in",train_test_path,"==\n")

    # # Performing preprocess
    # datalist_path = "data/"+dataset_name+"/datalist-"+naming_running+"/"
    # temp_split = datalist_path.split("_")
    # datalist_train_path = temp_split[0]+"_"+temp_split[1]+"_"+temp_split[2]

    # # Preprocessing for training because it only depends preprocess method
    # if not os.path.exists(datalist_train_path):
    #     os.makedirs(datalist_train_path)

    #     print("\n===== Preprocessing training dataset =====\n")
    #     datalist_train = main_preprocess(pre_method,data_train_idx,targets,out_path,NODE_TYPE_DICT)

    #     # save datalist train
    #     save_as_pickle(datalist_train,datalist_train_path+"datalist_train.pickle")
    # else:
    #     datalist_train = read_pickle_file(datalist_train_path+"datalist_train.pickle")

    # if not os.path.exists(datalist_path):
    #     print("\n== Preprocessing dataset ==\n")
    #     os.makedirs(datalist_path)

    #     # make perturbation configuration
    #     perturb_config = {
    #                             "perturb_type": perturb_type,
    #                             "perturb_target": perturb_target,
    #                             "k":k
    #                         }
        
    #     print("\n===== Preprocessing testing dataset original =====\n")
    #     datalist_test_original = main_preprocess(pre_method,data_test_idx,targets,out_path,NODE_TYPE_DICT)

    #     # save datalist test
    #     save_as_pickle(datalist_test_original,datalist_path+"datalist_test_original.pickle")

    #     # print("\n===== Preprocessing testing dataset perturbation =====\n")
    #     # datalist_test_perturbed = main_preprocess(pre_method,data_test_idx,targets,out_path,NODE_TYPE_DICT,perturb_config)

    #     # # save datalist test
    #     # save_as_pickle(datalist_test_perturbed,datalist_path+"datalist_test_perturbed.pickle")

    #     print("\n== Preprocessing are all finished ==\n")
    # else:
    #     datalist_test_original = read_pickle_file(datalist_path+"datalist_test_original.pickle")
    #     # datalist_test_perturbed = read_pickle_file(datalist_path+"datalist_test_perturbed.pickle")

    #     print("\n== Preprocessing dataset has been finished ==\n")

    # # Making data loader
    # train_loader = DataLoader(datalist_train, batch_size=batch_size)
    # test_original_loader = DataLoader(datalist_test_original, batch_size=batch_size)
    # # test_perturbed_loader = DataLoader(datalist_test_perturbed, batch_size=batch_size)
    
    # # Model setup. If it is the continuation of previous running, we use the previous model
    # if cont:
    #     model,criterion,optimizer,device = model_setup(model_name,learning_rate,weight_decay,cont_path)
    # else:
    #     model,criterion,optimizer,device = model_setup(model_name,learning_rate,weight_decay)

    # # Starting to training and testing
    # best_orig_loss = float('inf')
    # train_losses = []
    # train_accs = []
    # test_original_losses = []
    # test_original_accs = []
    # test_perturbed_losses = []
    # test_perturbed_accs = []
    # attack_success_rates = []

    # for epoch in range(num_epochs):
    #     # Training phase
    #     model, epoch_loss, epoch_acc = train(model,train_loader,criterion,optimizer,device)
    #     avg_loss = epoch_loss / len(train_loader)
    #     avg_acc = epoch_acc / len(train_loader)
        
    #     train_losses.append(avg_loss)
    #     train_accs.append(avg_acc)
        
    #     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')
        
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)

    #     # Test on original data
    #     orig_losses, orig_accs, orig_pre, orig_rec, orig_fsc, orig_true_labels, orig_pred_labels = test_with_prediction(
    #         model,test_original_loader,criterion,device)
    #     avg_orig_loss = sum(orig_losses) / len(orig_losses)
    #     avg_orig_acc = sum(orig_accs) / len(orig_accs)
    #     test_original_losses.append(avg_orig_loss)
    #     test_original_accs.append(avg_orig_acc)

    #     # # Test on perturbated data
    #     # pert_losses, pert_accs, pert_pre, pert_rec, pert_fsc, pert_true_labels, pert_pred_labels = test_with_prediction(
    #     #     model,test_perturbed_loader,criterion,device)
    #     # avg_pert_loss = sum(pert_losses) / len(pert_losses)
    #     # avg_pert_acc = sum(pert_accs) / len(pert_accs)
    #     # test_perturbed_losses.append(avg_pert_loss)
    #     # test_perturbed_accs.append(avg_pert_acc)

    #     # # Calculate attacke success rate
    #     # # ASR = % of correctly classified original samples that are misclassified after perturbation
    #     # correctly_classified_orig = [(i, label) for i, (pred, label) in enumerate(zip(orig_pred_labels, orig_true_labels)) if pred == label]
    #     # successful_attacks = 0
    #     # for idx, true_label in correctly_classified_orig:
    #     #     if pert_pred_labels[idx] != true_label:
    #     #         successful_attacks += 1

    #     # asr = successful_attacks / len(correctly_classified_orig) if len(correctly_classified_orig) > 0 else 0
    #     # attack_success_rates.append(asr)

    #     print(f'Original Test - Loss: {avg_orig_loss:.4f}, Accuracy: {avg_orig_acc:.4f}, '
    #       f'Precision: {orig_pre:.4f}, Recall: {orig_rec:.4f}, F1-Score: {orig_fsc:.4f}')
    #     # print(f'Perturbed Test - Loss: {avg_pert_loss:.4f}, Accuracy: {avg_pert_acc:.4f}, '
    #     #   f'Precision: {pert_pre:.4f}, Recall: {pert_rec:.4f}, F1-Score: {pert_fsc:.4f}')
    #     # print(f'Attack Success Rate: {asr:.4f}')
        
    #     if avg_orig_loss < best_orig_loss:
    #         best_orig_loss = avg_orig_loss
    #         torch.save(model, model_path+'model_'+naming_running+"_"+str(epoch+1+start_model_number)+'.pth')
    #         print("== Saved trained model at epoch:",epoch+1+start_model_number,"==")

    #     plot_and_save_training_metrics(train_losses,test_original_losses,train_accs,test_original_accs,"figure-test-"+naming_running)


if __name__ == "__main__":
    main()