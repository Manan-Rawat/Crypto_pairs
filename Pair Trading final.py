#!/usr/bin/env python
# coding: utf-8

# In[39]:


import glob
import os
import pandas as pd
from scipy.stats import rankdata
from statsmodels.api import add_constant
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import coint
import numpy as np
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton, Gumbel, Frank
from copulas.univariate import StudentTUnivariate
from copulas.visualization import scatter_2d
import time
path="/data//"
csv_files = glob.glob(os.path.join("*.csv"))
file_data_dict = {}
modified_dict = {}
dfl = []
for idx, file in enumerate(csv_files):
    file_name = os.path.basename(file)
    instrument_name = file_name.split('data_')[1].split('-USDT-SWAP.csv')[0]
    x = pd.read_csv(file, parse_dates=['time'], index_col='time')['close']
    dfl.append(x)
    modified_dict[idx] = instrument_name

merged_df = pd.concat(dfl, axis=1)
merged_df.columns = [modified_dict[i] for i in range(len(dfl))]
merged_df = merged_df.dropna()

cdf_df = merged_df.apply(lambda x: rankdata(x) / len(x), axis=0)

train_size = int(len(cdf_df) * 0.6)
train_data, test_data = cdf_df.iloc[:train_size], cdf_df.iloc[train_size:]

def mean_reversion_strategy(data, window, position_size=100, transaction_fee_rate=0.00015, slippage_rate=0.00015, k1=1.0, k2=1.0):
    mean_spread = data.rolling(window).mean()
    std_spread = data.rolling(window).std()
    position, entry_spread, cumulative_pnl, pnl_list = 0, 0, 0, []

    for i in range(len(data)):
        current_spread = data.iloc[i]
        upper_entry_threshold = mean_spread.iloc[i] + k1 * std_spread.iloc[i]
        lower_entry_threshold = mean_spread.iloc[i] - k2 * std_spread.iloc[i]
        upper_exit_threshold = mean_spread.iloc[i]
        lower_exit_threshold = mean_spread.iloc[i]

        if position == 0:
            if current_spread > upper_entry_threshold:
                position = -1
                entry_spread = current_spread
            elif current_spread < lower_entry_threshold:
                position = 1
                entry_spread = current_spread
        elif position == 1 and current_spread > lower_exit_threshold:
            pnl = (current_spread - entry_spread) * position_size
            transaction_cost = (transaction_fee_rate + slippage_rate) * (entry_spread + current_spread)
            pnl -= transaction_cost
            cumulative_pnl += pnl
            pnl_list.append(pnl)
            position = 0
        elif position == -1 and current_spread < upper_exit_threshold:
            pnl = (entry_spread - current_spread) * position_size
            transaction_cost = (transaction_fee_rate + slippage_rate) * (entry_spread + current_spread)
            pnl -= transaction_cost
            cumulative_pnl += pnl
            pnl_list.append(pnl)
            position = 0

    if position != 0:
        unrealized_pnl = (current_spread - entry_spread) * position_size if position == 1 else (entry_spread - current_spread) * position_size
        cumulative_pnl += unrealized_pnl
        pnl_list.append(unrealized_pnl)

    return cumulative_pnl, pnl_list

windows = {"15min": 3, "20 min": 4, "30 min": 6, "45 min": 9, "1hr": 12, "3hr": 36, "6hr": 72, "12hr": 144, "1day": 288, "1week": 2016}
results = []

for i, col1 in enumerate(cdf_df.columns):
    for j, col2 in enumerate(cdf_df.columns):
        if i >= j:
            continue

        pair_data = train_data[[col1, col2]].dropna()
        time_start = time.time()

        best_window, best_pnl, best_window_name = None, -np.inf, ""

        for window_name, window in windows.items():
            y, X = pair_data[col1], add_constant(pair_data[col2])
            rolling_model = RollingOLS(y, X, window=window).fit()
            spread = y - (rolling_model.params['const'] + rolling_model.params[col2] * X[col2])
            total_pnl, _ = mean_reversion_strategy(spread, window)

            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_window = window
                best_window_name = window_name

        time_taken = time.time() - time_start
        results.append((f"{col1}-{col2}", time_taken, best_pnl, best_window_name))
        print(f"PAIR {col1}-{col2}, TIME TAKEN - {time_taken:.2f}s, TOTAL PNL - {best_pnl:.2f}, BEST WINDOW - {best_window_name}")

top_20_pairs = sorted(results, key=lambda x: x[2], reverse=True)[:20]

top_20_cointegration_results = [
    (pair[0], coint(train_data[pair[0].split('-')[0]], train_data[pair[0].split('-')[1]])[1], pair[2], pair[3])
    for pair in top_20_pairs
]

print("\nTop 20 Pairs:")
for pair in top_20_cointegration_results:
    pair_name, cointegration_pval, best_pnl, best_window_name = pair
    print(f"PAIR: {pair_name}, PNL: {best_pnl:.2f}, COINTEGRATION P-VALUE: {cointegration_pval:.4f}, BEST WINDOW: {best_window_name}")

trade_success_results = []
s=0

for pair in top_20_pairs:
    col1, col2 = pair[0].split('-')
    pair_data = test_data[[col1, col2]].dropna()
    
    # Run Rolling OLS to estimate spread
    y, X = pair_data[col1], add_constant(pair_data[col2])
    rolling_model = RollingOLS(y, X, window=windows[pair[3]]).fit()
    spread = y - (rolling_model.params['const'] + rolling_model.params[col2] * X[col2])

    # Calculate mean reversion success rate
    success_rate = mean_reversion_trade_success(spread, windows[pair[3]])

    trade_success_results.append((pair[0], pair[3], pair[2], success_rate))

# Print Success Rates
print("\nMean Reversion Trade Success Rates:")
for result in trade_success_results:
    pair_name, best_window, best_pnl, success_rate = result
    print(f"PAIR: {pair_name}, BEST WINDOW: {best_window}, PNL: {best_pnl:.2f}, SUCCESS RATE: {success_rate:.2f}%")
    s=s+success_rate

print (s/20)


quantile_data = cdf_df

copulas = {
    'Gaussian': GaussianMultivariate,
    't-Copula': StudentTUnivariate,
    'Clayton': Clayton,
    'Gumbel': Gumbel,
    'Frank': Frank
}

fit_results = []

for i, col1 in enumerate(quantile_data.columns):
    for j, col2 in enumerate(quantile_data.columns):
        if i >= j:
            continue

        pair_data = quantile_data[[col1, col2]].dropna().values

        best_copula, best_aic, best_params = None, float('inf'), None

        for copula_name, CopulaClass in copulas.items():
            try:
                copula = CopulaClass()
                copula.fit(pair_data)
                log_likelihood = copula.log_likelihood(pair_data)
                num_params = len(copula.to_dict()['theta'])
                aic = 2 * num_params - 2 * log_likelihood

                if aic < best_aic:
                    best_aic = aic
                    best_copula = copula_name
                    best_params = copula.to_dict()

            except Exception as e:
                print(f"Failed to fit {copula_name} for pair {col1}-{col2}: {e}")

        fit_results.append({
            'pair': f"{col1}-{col2}",
            'copula': best_copula,
            'aic': best_aic,
            'params': best_params
        })

sorted_results = sorted(fit_results, key=lambda x: x['aic'])
top_5_pairs = sorted_results[:5]

print("Top 5 Fitted Copula Pairs:")
for result in top_5_pairs:
    print(f"Pair: {result['pair']}, Copula: {result['copula']}, AIC: {result['aic']}, Params: {result['params']}")

print("\nTrading Strategy Based on Conditional Probabilities:")
def trading_strategy_with_copula(pair_data, copula_params, copula_class, threshold=0.05):
    copula = copula_class()
    copula.from_dict(copula_params)

    trades = []
    for u1, u2 in pair_data:
        cond_prob_1 = copula.conditional_probability([u1, u2], given=1)
        cond_prob_2 = copula.conditional_probability([u1, u2], given=0)

        if cond_prob_1 <= threshold and cond_prob_2 >= 1 - threshold:
            trades.append("Long Spread")
        elif cond_prob_2 <= threshold and cond_prob_1 >= 1 - threshold:
            trades.append("Short Spread")
        else:
            trades.append("Exit")

    return trades

for result in top_5_pairs:
    pair = result['pair']
    copula_name = result['copula']
    params = result['params']
    pair_columns = pair.split('-')
    pair_data = quantile_data[pair_columns].dropna().values
    trades = trading_strategy_with_copula(pair_data, params, copulas[copula_name])

    print(f"Pair: {pair}, Trades: {trades}")



# In[ ]:




