import pandas as pd

def read(name_of_dataset):
    data_path = "./data/raw/" + name_of_dataset + ".csv"
    names_path = "./data/raw/" + name_of_dataset + "_names.txt"
    
    with open(names_path, 'r') as file:
        header = []
        for line in file:
            header.append(line.strip())
            
    df = pd.read_csv(data_path, header=None)
    df.columns = header


    categorical_path = "./data/raw/" + name_of_dataset + "_categorical.txt"
    numerical_path = "./data/raw/" + name_of_dataset + "_numerical.txt"
    
    with open(categorical_path, 'r') as file:
        categorical_vars = []
        for line in file:
            categorical_vars.append(line.strip())

    with open(numerical_path, 'r') as file:
        numerical_vars = []
        for line in file:
            numerical_vars.append(line.strip())

    return df, numerical_vars, categorical_vars