# calculate the transfer entropy
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import TE_calc


#calculate transfer entropy

def transfer_ent(df, lb_win):
    """_summary_

    Args:
        df (dataframe name): a dataframe with at leat two column to calculate transfer entropy between columns
        lb_win (integer): the value of lookback window(the history that te used)

    Returns:
        dataframe: it return an dataframe with for intancse rate(first column of df) to close flow(second col of df), 
        close to rate flow, mean flow
    """
    flow = TE_calc.create_traintarg(df, lb_win)
    flow_rate_ext = [el for el in flow if isinstance(el, list)]
    index_names = ['in_flow', 'out_flow']*(len(flow_rate_ext) //2)
    flow_rate_ext_df = pd.DataFrame(flow_rate_ext, index=index_names, columns=[str(df.columns[0]), str(df.columns[1])]) 
    flow_rate_ext_df = flow_rate_ext_df[flow_rate_ext_df.index != 'in_flow']
    flow_rate_ext_df['mean_flow'] = np.mean(flow_rate_ext_df[[df.columns[0], df.columns[1]]], axis=1) 
    return flow_rate_ext_df


def trans_concat(larger_df, df):
    """_summary_

    Args:
        larger_df (dataframe name): the name of larger dataframe (usually the dataframe that want to merge with transfer entropy dataframe)
        df (dataframe name): transfer entropy dataframe

    Returns:
        dataframe: the merged dataframe contain the two dataframe and transfer enteopy dataframe
    """
    row_to_remove = larger_df.shape[0] - df.shape[0]
    new_merged_df = larger_df.iloc[row_to_remove:, :]
    new_merged_df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    #now concate these two dataframe (as they are in the same shape)
    last_df = pd.concat([df['mean_flow'], new_merged_df], axis=1) 
    return last_df








