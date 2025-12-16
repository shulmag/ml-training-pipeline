'''
Description: Adds flags to trades to provide additional features.
'''
import numpy as np
import pandas as pd
import multiprocess as mp

from ficc.utils.auxiliary_variables import IS_REPLICA, IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR, REPLICA_COUNT, flatten
from ficc.utils.initialize_pandarallel import initialize_pandarallel


def subarray_sum(lst, target_sum, indices):
    '''The goal is to find a sublist in `lst`, such that the sum of the sublist equals 
    `target_sum`. If such a sublist cannot be formed, then return an dempty list. 
    Otherwise return the indices that should be removed from `lst` so that summing the 
    remaining items equals `target_sum`. The sublist in `indices` is returned. 
    Reference: https://www.geeksforgeeks.org/find-subarray-with-given-sum/ '''
    len_lst = len(lst)
    current_sum = lst[0]
    start, end = 0, 1
    while end <= len_lst:
        while current_sum > target_sum and start < end - 1:    # remove items from beginning of current sublist if the current sum is larger than the target
            current_sum -= lst[start]
            start += 1
        if current_sum == target_sum:
            return indices[start:end]
        if end < len_lst:
            current_sum += lst[end]
        end += 1
    return []


def subarray_sum_equaling_zero(lst, indices):
    '''Given an unsorted array of integers `lst`, find a subarray that adds to 0. 
    If there is more than one subarray return the longest one.
    Reference: https://www.geeksforgeeks.org/find-subarray-with-given-sum-in-array-of-integers/ '''
    d = dict()
    current_sum = 0
    left_idx_of_longest_subarray, right_idx_of_longest_subarray = None, None

    for idx in range(len(lst)):
        current_sum += lst[idx]
        if current_sum == 0:
            left_idx_of_longest_subarray, right_idx_of_longest_subarray = 0, idx
        if current_sum in d:
            potential_left_idx = d[current_sum] + 1
            potential_right_idx = idx
            if left_idx_of_longest_subarray is None or right_idx_of_longest_subarray - left_idx_of_longest_subarray < potential_right_idx - potential_left_idx:
                left_idx_of_longest_subarray, right_idx_of_longest_subarray = potential_left_idx, potential_right_idx
        if current_sum not in d: d[current_sum] = idx    # do not put the index into the map if it already exists, in order to have the largest list

    return indices[left_idx_of_longest_subarray : right_idx_of_longest_subarray + 1] if left_idx_of_longest_subarray is not None else []


def _select_apply_function(use_parallel_apply: bool):
    '''Choose between .apply(...) and .parallel_apply(...) for the groupby.'''
    if use_parallel_apply: initialize_pandarallel()
    return pd.core.groupby.GroupBy.parallel_apply if use_parallel_apply else pd.core.groupby.GroupBy.apply


def _add_same_day_flag_for_group(group_df):
    '''This flag denotes a non-DD trade where the dealer had the purchase and sell lined up 
    beforehand. We mark a trade as same day when the trade is part of a contiguous set of trades 
    such that dealer purchases add up to dealer sold. 
    Algorithmically, we set all the sell trades to have their par_traded values negated, and then 
    look for a subarray sum of zero for the modified par_traded values.'''
    if {'S', 'P'} != set(group_df['trade_type']): return []
    sell_mask = group_df['trade_type'] == 'S'
    par_traded_sell_negative_purchase_positive = group_df['par_traded'].to_numpy()
    par_traded_sell_negative_purchase_positive[sell_mask] = -1 * group_df[sell_mask]['par_traded']
    return subarray_sum_equaling_zero(par_traded_sell_negative_purchase_positive, group_df.index)


def add_same_day_flag_with_apply(df, flag_name=IS_SAME_DAY, use_parallel_apply=True):
    '''Call `_add_same_day_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.
    
    Note that using the groupby apply function is not working and 
    throwing: `TypeError: Series.name must be a hashable type`. Unable
    to figure out why this is the case, but seem to think it may be a 
    pandas error because the operation is performing fine with just 
    a for loop (see function below).'''
    df = df.astype({'par_traded': np.float64})    # `par_traded` type is Category so need to change it order to sum up; chose float64 to prevent potential rounding errors

    df[flag_name] = False
    group_by_day_cusip = df[df['trade_type'] != 'D'].groupby(['trade_date', 'cusip'], observed=True)[['par_traded', 'trade_type']]    # only need the 'par_traded' and 'trade_type' columns in the helper function
    day_cusip_to_indices_to_mark = _select_apply_function(use_parallel_apply)(group_by_day_cusip, _add_same_day_flag_for_group)
    indices_to_mark = day_cusip_to_indices_to_mark.sum()
    
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_same_day_flag(df, flag_name=IS_SAME_DAY, use_parallel_apply=True):
    '''Call `_add_same_day_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.
    
    Future work to speed up the for loop with multiple cores: 
    https://stackoverflow.com/questions/59184496/how-to-make-the-following-for-loop-use-multiple-core-in-python '''
    df = df.astype({'par_traded': np.float64})    # `par_traded` type is Category so need to change it order to sum up; chose float64 to prevent potential rounding errors

    df[flag_name] = False
    group_by_day_cusip = df[df['trade_type'] != 'D'].groupby(['trade_date', 'cusip'], observed=True)[['par_traded', 'trade_type']]    # only need the 'par_traded' and 'trade_type' columns in the helper function

    if use_parallel_apply:
        _add_same_day_flag_for_group_func = lambda _, sub_df: _add_same_day_flag_for_group(sub_df)
        with mp.Pool() as pool_object:
            indices_to_mark = pool_object.starmap(_add_same_day_flag_for_group_func, group_by_day_cusip)
    else:
        indices_to_mark = [_add_same_day_flag_for_group(sub_df) for _, sub_df in group_by_day_cusip]
    indices_to_mark = flatten(indices_to_mark)

    df.loc[indices_to_mark, flag_name] = True
    return df


def add_replica_flag(df, flag_name=IS_REPLICA):
    '''Mark a trade as a replica if there is a trade on the same 
    day with the same price, same direction, and same quantity. The idea 
    of marking these trades is to exclude them from the trade history, as 
    these trades are probably being sold in the same block, and so having 
    all of these trades in the trade history would be less economically 
    meaningful.'''
    group_by_day_cusip_quantity_price_tradetype = df.groupby(['trade_date', 'cusip', 'quantity', 'dollar_price', 'trade_type'], observed=True)
    df[flag_name] = group_by_day_cusip_quantity_price_tradetype['cusip'].transform('size')    # chose `.transform('size')` instead of `.transform(len)` since it is faster https://stackoverflow.com/questions/23017625/dataframe-add-column-with-the-size-of-a-group
    df.loc[:, flag_name] = df[flag_name] > 1
    return df


def add_bookkeeping_flag(df, flag_name=IS_BOOKKEEPING):
    '''Mark an inter-dealer trade as bookkeeping if there are multiple 
    inter-dealer trades of the same quantity at the same price for a 
    particular day.'''
    df_dd = df[df['trade_type'] == 'D']
    group_by_day_cusip_quantity_price_tradetype = df_dd.groupby(['trade_date', 'cusip', 'quantity', 'dollar_price'], observed=True)
    df_dd[flag_name] = group_by_day_cusip_quantity_price_tradetype['cusip'].transform('size')    # chose `.transform('size')` instead of `.transform(len)` since it is faster https://stackoverflow.com/questions/23017625/dataframe-add-column-with-the-size-of-a-group
    indices = np.where(df_dd[flag_name] > 1)    # chose `np.where` since it is the fastest way to perform this subprocedure: https://stackoverflow.com/questions/52173161/getting-a-list-of-indices-where-pandas-boolean-series-is-true
    indices_to_mark = df_dd.index[indices]

    df[flag_name] = False
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_replica_count_flag(df, flag_name=REPLICA_COUNT):
    '''This numerical flag denotes the number of trades with the same trade_date, 
    cusip, quantity, dollar_price, and trade_type that occur before the trade.'''
    group_by_day_cusip_quantity_price_tradetype = df.groupby(['trade_date', 'cusip', 'quantity', 'dollar_price', 'trade_type'], observed=True)[['cusip']]
    df[flag_name] = group_by_day_cusip_quantity_price_tradetype.cumcount(ascending=False)
    return df


def _add_ntbc_precursor_flag_for_group(group_df):
    '''This flag denotes an inter-dealer trade that occurs on the same day as 
    a non-transaction-based-compensation customer trade with the same price and 
    quantity. The idea for marking it is that this inter-dealer trade may not be 
    genuine (i.e., window-dressing). Note that we have a buffer of occurring on 
    the same day since we see examples in the data (e.g., cusip 549696RS3, 
    trade_datetime 2022-04-01) having the corresponding inter-dealer trade occurring 
    4 seconds before, instead of the exact same time, as the customer bought trade.'''
    is_dd_trade = group_df['trade_type'] == 'D'
    if (len(group_df) < 2) or (not group_df['is_non_transaction_based_compensation'].any()) or (not is_dd_trade.any()): return []
    return group_df[is_dd_trade].index.to_list()


def add_ntbc_precursor_flag(df, flag_name=NTBC_PRECURSOR, use_parallel_apply=True):
    '''Call `_add_ntbc_precursor_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.'''
    df[flag_name] = False
    group_by_day_cusip_quantity_price = df.groupby(['trade_date', 'cusip', 'quantity', 'dollar_price'], observed=True)[['is_non_transaction_based_compensation', 'trade_type']]    # only need the 'is_non_transaction_based_compensation' and 'trade_type' columns in the helper function
    day_cusip_quantity_price_to_indices_to_mark = _select_apply_function(use_parallel_apply)(group_by_day_cusip_quantity_price, _add_ntbc_precursor_flag_for_group)
    indices_to_mark = day_cusip_quantity_price_to_indices_to_mark.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df
