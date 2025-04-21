
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import config

FF48_DAILY = './data/48_Industry_Portfolios_Daily.csv'
FF48_MONTHLY = './data/48_Industry_Portfolios_Monthly.csv'
DASH_STYLES = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]

def get_trimmed_data():
    """
    Load and trim the data.
    Returns:
        pd.DataFrame: The trimmed data.
    """
    if config.get('data.dataset') == 'FF48':
        data = pd.read_csv(FF48_DAILY, index_col=0)

    window_size = config.get('parameter.window_size')
    start_date = config.get('data.period.start')
    end_date = config.get('data.period.end')

    if config.get('data.compensate'):
        start_indices = data.index[data.index.astype(str).str.startswith(start_date)]
        end_indices = data.index[data.index.astype(str).str.startswith(end_date)]
        if not start_indices.empty:
            start_match = start_indices[0]
            start_loc = data.index.get_loc(start_match)
        if not end_indices.empty:
            end_match = end_indices[-1]
            end_loc = data.index.get_loc(end_match)
        trimmed_data = data.iloc[start_loc - window_size : end_loc + 1]
    else:
        trimmed_data = data.iloc[(start_date <= data.index.values.astype(str)) & (data.index.values.astype(str) <= end_date)]

    # Convert the index to datetime format
    trimmed_data.index = pd.to_datetime(trimmed_data.index, format='%Y%m%d')

    return trimmed_data


def plot_full_cumulative_wealth(results):
    """
    Plot the cumulative wealth of different algorithms over the full period.
    Parameters:
        results (dict): Dictionary containing the cumulative wealth of each algorithm.
    """
    period = f"{config.get('data.period.start')[:4]}-{config.get('data.period.end')[:4]}"
    sns.set_theme(style='whitegrid', font_scale=3.2)
    plt.figure(figsize=(60, 25))
    ax = sns.lineplot(data=results, palette='bright', linewidth=2.7, dashes=DASH_STYLES)
    ax.set(xlabel='Date', ylabel='Cumulative Wealth', title=f"{config.get('data.dataset')} Algorithm Comparison ({period})");

    ax.figure.savefig(config.get('plot.output_dir') + '/cw_overall.png')


def plot_period_cumulative_wealth(results_pct):
    """
    Plot the cumulative wealth of different algorithms.
    Parameters:
        results (dict): Dictionary containing the cumulative wealth of each algorithm.
    """
    num_algorithms = len(results_pct.columns)
    # Determine the number of plots needed
    # extract the year from the start date and end date, e.g., "198501" -> 1985
    start_year = config.get('data.period.start')[:4]
    end_year = config.get('data.period.end')[:4]
    start_year_int = int(start_year)
    end_year_int = int(end_year)

    # divide the years into 10-year intervals    
    interval = config.get('plot.interval')
    num_plots = (end_year_int - start_year_int + 1) / interval
    # get integer, 3.0 -> 3, 3.5 -> 4
    num_plots = int(num_plots) if num_plots.is_integer() else int(num_plots) + 1

    # calculate the wealth for each interval
    wealths = {}
    for i in range(num_plots):
        start = str(start_year_int + i * interval)
        end = str(min(start_year_int + (i+1) * interval - 1, end_year_int))

        wealth = np.cumprod(pd.DataFrame(np.concatenate(
            (np.ones(num_algorithms).reshape(-1, num_algorithms), 
             results_pct.loc[start:end, :].values+1), axis=0), 
             columns=results_pct.columns, 
             index=pd.date_range(start=results_pct.loc[start:end, :].index[0]-pd.Timedelta("1d"), 
                             end=results_pct.loc[start:end, :].index[-1], 
                             periods=results_pct.loc[start:end, :].shape[0]+1).date))

        wealths.update({f"{start_year}-{end_year}": wealth})

    sns.set_theme(style='whitegrid', font_scale=3.3)

    for period, wealth_period in wealths.items():
        plt.figure(figsize=(60, 25))
        ax = sns.lineplot(data=wealth_period, palette="bright", linewidth=2.7, dashes=DASH_STYLES)
        ax.set(xlabel='Date', ylabel='Cumulative Wealth', title=f"{config.get('data.dataset')} Algorithm Comparison ({period})")
        ax.figure.savefig(config.get('plot.output_dir') + '/cw_{period}.png')
        plt.close()

    
