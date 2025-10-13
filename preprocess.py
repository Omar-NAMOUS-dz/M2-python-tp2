import cfg
import pandas as pd
from read import read
from exploration import get_missing_ratio, get_chi_2_matrix, get_outliers

def set_variables_types(data, numerical_vars, categorical_vars):
    df = data.copy()

    df[categorical_vars] = df[categorical_vars].astype('category')

    for num in numerical_vars:
        df[num] = pd.to_numeric(df[num], downcast='float', errors='coerce') 

    return df

def delete_columns_with_missing(df, missing_treshold=0.85):
    ratio = get_missing_ratio(df)
    to_delete = []
    for c in df.columns:
        if ratio[c] > missing_treshold:
            to_delete.append(c) 
    df = df.drop(to_delete, axis=1)
    return df, to_delete

    
def delete_duplicate_rows(df):
    df = df.drop_duplicates()
    return df


def replace_missing(df, numerical_vars, categorical_vars):
    for j in df.columns:
        if j in numerical_vars:
            mean = df[j].mean()
            df.loc[df[j].isin(cfg.MISSING_VALUES), j] = mean
        if j in categorical_vars:
            mode = df[j].mode()[0]
            df.loc[df[j].isin(cfg.MISSING_VALUES), j] = mode
    return df

def group_in_other(df, categorical_vars, grouping_treshold=0.01):
    for j in df.columns:
        if j in categorical_vars:
            freq = df[j].value_counts()
            for i in df.index:
                value = df.at[i, j]
                if freq[value] < grouping_treshold:
                    df.at[i, j] = "Other"
    return df

def delete_highly_correlated_numeric(df, numerical_vars, pearson_treshold=0.95):
    corr = df[numerical_vars].corr(method='pearson', numeric_only=True).abs()
    
    deleted = {v: False for v in numerical_vars}

    for v1 in numerical_vars:
        if not deleted[v1]:
            for v2 in numerical_vars:
                if v1 != v2 and not deleted[v2] and corr[v1][v2] > pearson_treshold:
                    deleted[v1] = True

    df = df.drop(columns=[v for v in numerical_vars if deleted[v]])
    numerical_vars = [v for v in numerical_vars if not deleted[v] ]
    return df, numerical_vars
    

def delete_highly_correlated_categoric(df, categorical_vars, chi_2_treshold=1e6):
    chi2_stat_matrix, pval_matrix = get_chi_2_matrix(df, categorical_vars)

    deleted = {v: False for v in categorical_vars}

    for v1 in categorical_vars:
        if not deleted[v1]:
            for v2 in categorical_vars:
                if v1 != v2 and not deleted[v2] and chi2_stat_matrix[v1][v2] > chi_2_treshold:
                    deleted[v1] = True

    df = df.drop(columns=[v for v in categorical_vars if deleted[v]])
    categorical_vars = [v for v in categorical_vars if not deleted[v] ]
    return df, categorical_vars


def delete_outliers(df, numerical_vars):
    for v in numerical_vars:
        outliers = get_outliers(df[v])
        to_delete = [outlier["sample index"] for outlier in outliers]
        df = df.drop(df.index[to_delete])
        df = df.reset_index(drop=True)

    return df
                
def one_hot_encoding(df, categorical_vars):
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    return df
