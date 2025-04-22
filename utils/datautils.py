import pickle
import pandas as pd

def read_data(path):
    return pd.read_json(path)

def select(dataset):
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]

    result_neg = result[result["target"]==0].head(200)
    result_pos = result[result["target"]==1].head(200)

    comb_result = pd.concat([result_neg,result_pos]).sort_index()

    print("Dataset filtered for FFmpeg project only with maximum length 1200")
    print("We only took 200 for each target for efficiency")
    return comb_result

def make_file(raw,path):
    for idx, row in raw.iterrows():
        file_name = f"{idx}.c"
        with open(path + file_name, 'w') as f:
            f.write(row.func)
    print("Finish write function to C files")

def save_as_pickle(ob,file_name):
    with open(file_name,"wb") as file:
        pickle.dump(ob, file)

def read_pickle_file(file_name):
    with open(file_name,"rb") as file:
        return pickle.load(file)

def prepare_data(path,dataset_name):
    print("\n== Preparing dataset:",dataset_name,"==\n")

    if dataset_name == "self-devign":
        raw_path = "./data/self-devign/function.json"

        # read raw dataset
        raw = read_data(raw_path)
        print(raw)
        print(raw.columns)

        dataset = select(raw)
        print(dataset)
    
    elif dataset_name == "devign":
        raw_path = "./data/devign/devign_duplicate.json"

        # read raw dataset
        dataset = read_data(raw_path)
        print(dataset)
        print(dataset.columns)

    else:
        sys.exit("\n== There is no such dataset name ==\n")

    return dataset