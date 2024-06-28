import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau,t
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEFAULT_PROP_NAMES = ['LogBCF', 'LogP', 'LogVP', 'MP', 'LogKOC', 'BP', 'LogHL', 'Clint', 'FU',
       'LogOH', 'LogKmHL', 'LogKOA', 'LogMolar', 'LogHalfLife']
DEFAULT_PRED_NAMES = ['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9', 'pred_10', 'pred_11', 'pred_12', 'pred_13']


def plot_comparison(single_task_dfs, multi_task_df, properties=DEFAULT_PROP_NAMES, multi_predictions=DEFAULT_PRED_NAMES):
    """
    Plot a comparison of single-task and multi-task predictions for multiple properties.

    Args:
        single_task_dfs (list): A list of pandas DataFrames containing single-task predictions for each property.
        multi_task_df (pandas DataFrame): A pandas DataFrame containing multi-task predictions.
        properties (list, optional): A list of property names to plot. Defaults to DEFAULT_PROP_NAMES.
        multi_predictions (list, optional): A list of prediction column names for multi-task predictions.
            Defaults to DEFAULT_PRED_NAMES.

    Returns:
        None
    """
    rows = len(properties)
    cols = 2  # One for single-task, one for multi-task

    fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 6))  # Adjust the figure size as needed
    if rows == 1:
        axs = np.array([axs])  # Ensure axs is 2D array even for a single row

    for i, prop in enumerate(properties):
        # Prepare data for single-task
        single_task_df = single_task_dfs[i]
        valid_single = single_task_df[['SMILES', prop, 'pred_0']].dropna()
        true_single_values = valid_single[prop]
        pred_single_values = valid_single['pred_0']

        # Prepare data for multi-task
        valid_multi = multi_task_df[['SMILES', prop, multi_predictions[i]]].dropna()
        true_multi_values = valid_multi[prop]
        pred_multi_values = valid_multi[multi_predictions[i]]

        # Plotting single-task
        plot_scatter_and_line(axs[i][0], true_single_values, pred_single_values, f'Single-task: {prop}')

        # Plotting multi-task
        plot_scatter_and_line(axs[i][1], true_multi_values, pred_multi_values, f'Multi-task: {prop}')

    plt.tight_layout()
    plt.show()

def plot_scatter_and_line(ax, true_values, pred_values, title):
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    mae = mean_absolute_error(true_values, pred_values)
    r, _ = pearsonr(true_values, pred_values)
    r2 = r2_score(true_values, pred_values)
    tau, _ = kendalltau(true_values, pred_values)
    textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nPearson r: {r:.2f}\n$R^2$: {r2:.2f}\nKendall tau: {tau:.2f}'

    # Plot
    ax.scatter(true_values, pred_values, alpha=0.5)
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')
    ax.set_title(title)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    ax.set_aspect('equal', 'box')


def confidence_interval(data, confidence=0.90):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = sem * t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - margin_of_error, mean + margin_of_error

def calculate_statistics(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r, _ = pearsonr(true_values, predicted_values)
    tau, _ = kendalltau(true_values, predicted_values)
    return rmse, mae, r, tau

def create_scatter_plot(ax, true_values, predicted_values, title,c='skyblue'):
    xlim = ylim = (1, 5)
    ax.scatter(true_values, predicted_values, edgecolor='black', color=c)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.plot(xlim, ylim, 'k--')

    rmse, mae, r,tau = calculate_statistics(true_values, predicted_values)
    textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nPearson r: {r:.2f}\nKendall tau: {tau:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
