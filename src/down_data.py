import numpy as np
import pandas as pd
import datetime as dt
import talib
from typing import Callable
from typing import Union, Tuple

from fredapi import Fred
fred = Fred(api_key='fdf7fbad89ee632fd8abe48fed980983') 

import yfinance as yf

def fred_down(series_id, col1, col2, start, end, log=False):
    """_summary_

    Args:
        series_id (string): the name of fred symbol
        col1 (string): the name of date column, default: date
        col2 (string): the name of specific column to extract, default: rate
        start (string): start date 
        end (string):  end date
        log (bool, optional): if true calculate the log return of rate column. Defaults to False.

    Returns:
        dataframe: the dataframe of mentioned series_id
    """
    data = fred.get_series(series_id) 
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    data.rename(columns={"index": col1, 0:col2}, inplace=True) 
    data = data.loc[(data[col1] >= start) & (data[col1]<=end)] 
    if log==True:
        data['log_ret'] = np.log(data[col2]) - np.log(data[col2].shift(1))
        data= data.fillna(method='bfill')
        data = pd.DataFrame().assign(date=data['date'], rate=data['log_ret'])
    return data


def yfin_down(series_id, start, end, interval, log=False, *useless_cols):
    """_summary_

    Args:
        series_id (string): the name of yahoo finance history table
        start (string): _description_
        end (string): _description_
        interval (string): Valid intervals: 1m(minute),2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo(month),3mo
        log (bool, optional): if true calculate log return of Close colum. Defaults to False.
        useless_cols: there are some columns in yahoo fin history table that are unnecessary 

    Returns:
        dataframe: the dataframe of mentioned series_id
    """
    obj = yf.Ticker(series_id)
    data = obj.history(interval=interval, start=start, end=end) 
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.tz_localize(None) 
    # data = pd.DataFrame().assign(date=data['Date'], Close=data['Close'])
    data.rename(columns={"Date": "date"}, inplace=True)
    if log==True:
        data['Close'] = np.log(abs(data['Close'])) - np.log(abs(data['Close'].shift(1)))
        data = data.fillna(method='bfill') # type: ignore
    data = pd.DataFrame().assign(date=data['date'], Close=data['Close']) 
    # data = pd.DataFrame().assign(date=data['date'], Close=data['Close'], Open= data['Open'], High=data['High'], Low= data['Low']) 
    return data


def extend_concat(df1, df2, scale=''): 
    """concatenate fred dataframe(df1) with yahoo finance dataframe(df2) and return a dataframe with date, rate, and Close columns

    Args:
        df1 (string): fred dataframe, as out merge how parameter fixed 'left'
        df2 (string): yahoo finance dataframe 
        scale (str, optional): if its value is "M", then extend fred df from Month interval to day interval else 
        remain default df with month frequency. Defaults to ''.

    Returns:
        dataframe: _description_
    """
    if scale == 'M':
        df1['month'] = df1['date'].dt.to_period(scale)
        df2['month'] = df2['date'].dt.to_period(scale)
        merged_df = pd.merge(df1, df2, on='month', how='left')
        merged_df = pd.DataFrame().assign(date=merged_df['date_y'], rate=merged_df['rate'], Close=merged_df['Close'])
    else:
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        merged_df = pd.concat([df1, df2], axis=1)
        merged_df.reset_index(inplace=True)
        merged_df = pd.DataFrame().assign(date=merged_df['date'], rate=merged_df['rate'], Close=merged_df['Close']) 
    merged_df.set_index(['date'], inplace=True)
    return merged_df

def concat(data, data1):
    """_summary_

    Args:
        data (_type_): other yahoo finance dataframes
        data1 (_type_): forex dataframe

    Returns:
        _type_: merge two yahoo finance dataframes 
    """

    merged_df = pd.merge(data, data1, on='date', how='right').fillna(method="bfill")   # type: ignore 
    merged_df.set_index('date', inplace=True)
    return merged_df 
     

def merge_dataframes(dataframes):
  # Start with the last DataFrame as the base
    result = dataframes[-1].copy()
    
    # Iterate through the other DataFrames in reverse order
    for i in range(len(dataframes) - 2, -1, -1):
        df = dataframes[i]
        # Create a suffix for this DataFrame
        suffix = f'_{i+1}'
        
        # Rename columns in the current DataFrame, except 'date'
        rename_dict = {col: f'{col}{suffix}' for col in df.columns if col != 'date'}
        df_renamed = df.rename(columns=rename_dict)
        
        # Merge
        result = pd.merge(df_renamed, result, on='date', how='right').fillna(method="bfill")   # type: ignore 
    
    return result



def apply_talib_indicator(df: pd.DataFrame, 
                          indicator_func: Callable, 
                          input_columns: Union[str, Tuple[str, ...]], 
                          output_column: Union[str, Tuple[str, ...]], 
                          **kwargs) -> pd.DataFrame:
    """
    Apply a TA-Lib indicator function to a DataFrame and add the result as a new column.

    Parameters:
    df (pd.DataFrame): Input DataFrame with OHLCV data.
    indicator_func (Callable): TA-Lib indicator function to apply.
    input_columns (str or tuple of str): Column name(s) to use as input for the indicator function.
    output_column (str or tuple of str): Name(s) for the output column(s).
    **kwargs: Additional keyword arguments to pass to the indicator function.

    Returns:
    pd.DataFrame: DataFrame with the new indicator column(s) added.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Prepare input data
    if isinstance(input_columns, str):
        input_data = df_copy[input_columns].values
    else:
        input_data = [df_copy[col].values for col in input_columns]

    # Apply the indicator function
    result = indicator_func(input_data, **kwargs)

    # Add the result to the DataFrame
    if isinstance(output_column, str):
        df_copy[output_column] = result
    else:
        for i, col in enumerate(output_column):
            df_copy[col] = result[i]

    return df_copy


# df_sma = apply_talib_indicator(yah_df, talib.SMA, 'Close', 'SMA_20', timeperiod=20)   # type: ignore
# df_macd = apply_talib_indicator(yah_df, talib.MACD, 'close', ('MACD', 'MACD_signal', 'MACD_hist'),  # type: ignore
#                                 fastperiod=12, slowperiod=26, signalperiod=9) 
# df_rsi = apply_talib_indicator(yah_df, talib.RSI, 'close', 'RSI_14', timeperiod=14) #type: ignore
# df_atr = apply_talib_indicator(yah_df, talib.ATR, ('High', 'Low', 'Close'), 'ATR_14', timeperiod=14) # type: ignore 
