"""
General python utility functions
"""
import pandas as pd


def get_daily_vol(close, lookback=100):
    """
    Snippet 3.1, page 44, Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.

    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: Closing prices
    :param lookback: lookback period to compute volatility
    :return: series of daily volatility value
    """
    # Find index positions from 1 day ago
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]

    # Get previous close prices by position (use iloc with positions adjusted)
    # df0 contains positions, so df0-1 gives the previous day's position
    prev_positions = df0 - 1

    # Align with current index: take the last len(df0) elements of close
    current_index = close.index[close.shape[0] - df0.shape[0]:]

    # Calculate daily returns using position-based indexing
    current_close = close.loc[current_index]
    prev_close = close.iloc[prev_positions].values
    daily_ret = current_close / prev_close - 1

    # Apply EWM standard deviation
    df0 = daily_ret.ewm(span=lookback).std()
    return df0
