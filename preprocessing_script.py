from preprocess import *
from read import read
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset', default='census_income')
    
    args = parser.parse_args()
    df, numerical_vars, categorical_vars = read(args.dataset)
    df = set_variables_types(df, numerical_vars, categorical_vars)
    df, deleted = delete_columns_with_missing(df)
    categorical_vars = list(set(categorical_vars) - set(deleted))
    numerical_vars = list(set(numerical_vars) - set(deleted))
    df = replace_missing(df, numerical_vars, categorical_vars)
    df = delete_duplicate_rows(df)
    df, numerical_vars = delete_highly_correlated_numeric(df, numerical_vars)
    df, categorical_vars = delete_highly_correlated_categoric(df, categorical_vars)
    df = group_in_other(df, categorical_vars, grouping_treshold=0.01)
    df = one_hot_encoding(df, categorical_vars)

    df.to_csv('./data/processed/' + args.dataset + '_processed.csv', index=False)

    with open('./data/processed/' + args.dataset + '_names.txt', 'w') as f:
        f.writelines(f"{item}\n" for item in list(df.columns))
        
    with open('./data/processed/' + args.dataset + '_categorical.txt', 'w') as f:
        f.writelines(f"{item}\n" for item in categorical_vars)
        
    with open('./data/processed/' + args.dataset + '_numerical.txt', 'w') as f:
        f.writelines(f"{item}\n" for item in numerical_vars)