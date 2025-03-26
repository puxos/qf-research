
import matplotlib.pyplot as plt
import pandas as pd
# import pandas_datareader.data as web
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from functools import partial
import talib
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, kurtosis, skew

from portfolio import BanditPortfolio

data_daily = pd.read_csv('/Users/frank/Downloads/48_Industry_Portfolios_Daily.csv', index_col=0)
data_monthly = pd.read_csv('/Users/frank/Downloads/48_Industry_Portfolios.csv', index_col=0)

data_daily = data_daily[('197401' <= data_daily.index.values.astype(str)) & (data_daily.index.values.astype(str) <= '201912')]
data_monthly = data_monthly[('197401' <= data_monthly.index.values.astype(str)) & (data_monthly.index.values.astype(str) <= '201912')]

data_daily.index = pd.to_datetime(data_daily.index, format="%Y%m%d")
data_monthly.index = pd.to_datetime(data_monthly.index, format="%Y%m")

date_daily = data_daily.index.date
date_monthly = data_monthly.index.date

# The rate of return matrix, fill missing values with 100%
R = data_daily.values.T.astype(float)
R[R < -99]
R = (R + 100) / 100  # Gross Return

data_daily.head()

# Calcalation
window_size = 120
orthogonal_bandit_portfolio = BanditPortfolio(R)

orthogonal_bandit_portfolio.UCB(window_size = window_size)
ucb_ret = orthogonal_bandit_portfolio.reward

orthogonal_bandit_portfolio.TS(window_size = window_size)
ts_ret = orthogonal_bandit_portfolio.reward
mv_ret = orthogonal_bandit_portfolio.mv_reward

orthogonal_bandit_portfolio.PSR(window_size = window_size)
psr_ret = orthogonal_bandit_portfolio.reward

orthogonal_bandit_portfolio.UCBPSR(window_size = window_size)
ucbpsr_ret = orthogonal_bandit_portfolio.reward

#Baseline
constant_weight_rebalance = np.cumprod(R[:,window_size:].mean(axis=0))
equal_weight_portfolio = np.mean(np.cumprod(R[:,window_size:], axis=1), axis=0)

date = data_daily.index.values[window_size:]

ew_ret = np.mean(R[:,window_size:], axis=0)

mv_ret.shape

all_rets = {"MVP": mv_ret, "EW": ew_ret, "UCB1": ucb_ret, "TS": ts_ret, "MaxPSR": psr_ret, "PW-UCB1": ucbpsr_ret} 

z = pd.DataFrame(all_rets)-1

z.to_csv("new_rets.csv")

all_wealth = {"MVP": mv_wealth, "CWR": constant_weight_rebalance, "EW": equal_weight_portfolio, "UCB1": ucb_wealth, "TS": ts_wealth, "MaxPSR": psr_wealth, "PW-UCB1": ucbpsr_wealth} 