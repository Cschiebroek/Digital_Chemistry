import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kendalltau, pearsonr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def plot_property_histograms(df):
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    for i, (col, ax) in enumerate(zip(df.columns[1:], axes.flatten())):
        sns.histplot(df[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.show()

def plot_scatters(df_preds,df_ys):
    for prop in df_preds.columns:
        if prop == 'smiles':
            continue
    #make tmp df, where all nan are dropped for that property
        tmp_df = pd.DataFrame([df_preds[prop].rename("preds"),df_ys[prop].rename("ys")]).T
        tmp_df = tmp_df.dropna()
        kendall_tau,rmse_overall,mae_overall,within_03_overall,within_1_overall = get_stats(tmp_df["preds"],tmp_df["ys"])
        r2 = r2_score(tmp_df["ys"],tmp_df["preds"])
        pearson = pearsonr(tmp_df["ys"],tmp_df["preds"])[0]
        print(f'Property: {prop}')
        print(f'kendall_tau: {kendall_tau}')
        print(f'rmse_overall: {rmse_overall}')
        print(f'mae_overall: {mae_overall}')
        print(f'within_03_overall: {within_03_overall}')
        print(f'within_1_overall: {within_1_overall}')
        print(f'R2: {r2}')
        print(f'Pearson: {pearson}')
        plt.figure(figsize=(10, 10))
        plt.scatter(tmp_df["ys"], tmp_df["preds"], alpha=0.5)
        min_val = min(min(tmp_df["ys"]),min(tmp_df["preds"]))
        max_val = max(max(tmp_df["ys"]),max(tmp_df["preds"]))
        #add diagonal, and lines at +- 0.3 and +- 1
        plt.plot([min_val-1,max_val+1],[min_val-1,max_val+1],color='black')
        plt.plot([min_val-1,max_val+1],[min_val-1+0.3,max_val+1+0.3],color='black',linestyle='--')
        plt.plot([min_val-1,max_val+1],[min_val-1-0.3,max_val+1-0.3],color='black',linestyle='--')
        plt.plot([min_val-1,max_val+1],[min_val-1+1,max_val+1+1],color='black',linestyle='--')
        plt.plot([min_val-1,max_val+1],[min_val-1-1,max_val+1-1],color='black',linestyle='--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(prop)
        plt.show()

def get_stats(out_list,y_list):
    kendall_tau = kendalltau(y_list, out_list)[0]
    rmse_overall = np.sqrt(np.mean((np.array(y_list) - np.array(out_list)) ** 2))
    mae_overall = np.mean(np.abs(np.array(y_list) - np.array(out_list)))
    within_03_overall = np.mean(np.abs(np.array(y_list) - np.array(out_list)) < 0.3)
    within_1_overall = np.mean(np.abs(np.array(y_list) - np.array(out_list)) < 1)

    return kendall_tau,rmse_overall,mae_overall,within_03_overall,within_1_overall