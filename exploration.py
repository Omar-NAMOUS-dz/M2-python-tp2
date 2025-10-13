import cfg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency


pd.options.display.float_format = '{:.2f}'.format

def get_missing_ratio(df):
    ratio = {}
    for c in df.columns:
        missing = 0.0
        for n in df[c]:
            if n in cfg.MISSING_VALUES or pd.isna(c): missing += 1.0
        ratio[c] = float(missing)/float(len(df))
    return ratio

def get_general_info(df):
    num_samples = len(df)
    num_features = len(df.columns)
    var_type = {}
    for var in df.columns:
        if df[var].dtype != "float32" and df[var].dtype != "float64":
            var_type[var] = "Categorical"
        else:
            var_type[var] = "Numerical"

    numerical_vars = [v for v in var_type if var_type[v]=="Numerical"]
    categorical_vars = [v for v in var_type if var_type[v]=="Categorical"]

    return num_samples, num_features, numerical_vars, categorical_vars

def get_numerical_info(column):
    return column.describe()

def get_categorical_info(column):
    num_modalities = len(column.unique())
    modalities = list(pd.unique(column))
    freq = column.value_counts()
    return num_modalities, modalities, freq
    

    
def plot_numerical_distribution(df, column_name):
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    sns.stripplot(x=df[column_name], ax=ax1, color='blue', jitter=True, size=6)
    ax1.set_title(f'Distribution of {column_name}')
    ax1.set_ylabel('')
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    sns.boxplot(x=df[column_name], ax=ax2, color='lightcoral')
    ax2.set_yticks([])

    ax1.set_xlabel('')
    ax2.set_xlabel('Value')
    
    plt.show()

    

def plot_categorical_distribution(df, column_name):

    column = df[[column_name]]
    order = df[column_name].value_counts().index

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=column_name, data=column, palette='Set2', order=order)
    ax.set_xticklabels([])
    
    plt.title(f'Frequency of {column_name} Categories')
    plt.ylabel('Count')
    
    plt.show()

def get_outliers(column):
    median = column.median()
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    
    outliers = []
    i = 0
    for c in column:
        if c > median + 1.5 * (q3 - q1) or c < median - 1.5 * (q3 - q1):
            outliers.append({
                "sample index": i,
                "value": c
            })
        i += 1

    return outliers
        
def plot_pearson_correlation_matrix(df):
    corr = df.corr(method='pearson')

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=.5)
    
    plt.title('Pearson Correlation Matrix')
    plt.show()

def get_chi_2_matrix(df, categorical_vars):
    cols = categorical_vars
    chi2_stat_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                chi2_stat_matrix.loc[col1, col2] = np.nan
                pval_matrix.loc[col1, col2] = np.nan
            else:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                chi2_stat_matrix.loc[col1, col2] = chi2
                pval_matrix.loc[col1, col2] = p

    return chi2_stat_matrix, pval_matrix
    
def plot_chi_2_matrix(df, categorical_vars):

    chi2_stat_matrix, pval_matrix = get_chi_2_matrix(df, categorical_vars)

    with np.errstate(divide='ignore'):
        modified_pvals = 1.0 / (pval_matrix.astype(float) + 1.0)

    plt.figure(figsize=(16,12))
    sns.heatmap(chi2_stat_matrix.astype(float), annot=False, cmap='YlGnBu', square=True)
    plt.title('Chi-squared Test Statistic Matrix')
    plt.show()
    
    # Plot heatmap of -log10(p-values)
    plt.figure(figsize=(8,6))
    sns.heatmap(modified_pvals.astype(float), annot=False, cmap='coolwarm',vmin=0.99, vmax=1.0, square=True, cbar_kws={'label':'modified p-value'})
    plt.title('Significance (p-values) of Chi-squared Tests')
    plt.show()

    return chi2_stat_matrix
        

    