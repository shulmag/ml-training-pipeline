'''
 '''
import numpy as np
import pandas as pd


def _get_most_recent_index_and_others(df, with_alternative_trading_system_flag=False):
    '''If `with_alternative_trading_system_flag` is `True`, then return the most recent 
    index with the alternative trading system flag. If no trades in `df` have the 
    flag, then behave as if `with_alternative_trading_system_flag` is `False`. Note: 
    only inter-dealer trades can have the alternative trading system flag.'''
    if with_alternative_trading_system_flag:    # check whether there is a most recent index with the alternative trading system flag
        df_with_alternative_trading_system_flag = df[df['is_alternative_trading_system']]
        indices = df_with_alternative_trading_system_flag.index.to_list()
    if not with_alternative_trading_system_flag or indices == []:    # if the alternative trading system flag was not desired or not found
        indices = df.index.to_list()
    most_recent_index = indices[0]    # since `df` is sorted in descending order of `trade_date`, the first item is the most recent

    return most_recent_index, [index for index in df.index.to_list() if index != most_recent_index]


def _combine_trades(df, group_df):
    most_recent_trade_index, indices_to_remove = _get_most_recent_index_and_others(group_df)
    new_total_quantity = np.log10(sum(10 ** group_df['quantity']))    # undo log10 transformation before sum and reapply log10 transformation after sum
    df.loc[most_recent_trade_index]['quantity'] = new_total_quantity
    return df, indices_to_remove


def combine_and_remove_trades(df, verbose=False):
    if verbose: df_original = df.copy()    # df_original used only for printing progress at the end

    FEATURES_OF_INTEREST = ['cusip', 'quantity', 'dollar_price', 'trade_datetime', 'trade_type']
    IDENTIFIERS = ['cusip', 'rtrs_control_number']
    FLAGS = ['is_non_transaction_based_compensation', 'brokers_broker', 'is_lop_or_takedown', 'is_alternative_trading_system']
    FLAGS_TO_FILTER_ON = [flag for flag in FLAGS if flag not in ['is_alternative_trading_system']]    # this removes flags that we do not want to group on
    ALL_IMPORTANT_FEATURES = list(set().union(FEATURES_OF_INTEREST + IDENTIFIERS + FLAGS))

    df_select_features = df[ALL_IMPORTANT_FEATURES]
    df_select_features['brokers_broker'] = df_select_features['brokers_broker'].astype('string').fillna('none')    # replace the NaN value with 'none' so that we can use groupby (groupby doesn't work for NaN even with the dropna flag)
    groups_same_day_quantity_price_cusip_tradetype_flags = df_select_features.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'quantity', 'dollar_price', 'cusip', 'trade_type'] + FLAGS_TO_FILTER_ON)
    groups_same_day_quantity_price_cusip_tradetype_flags = {group_key: group_df for group_key, group_df in groups_same_day_quantity_price_cusip_tradetype_flags if len(group_df) > 1}    # removes groups with only 1 item since this is not really a group
    if verbose: print(f'Number of groups: {len(groups_same_day_quantity_price_cusip_tradetype_flags)}')

    groups_same_day_quantity_price_cusip_tradetype_flags_nonDD = {group_key: group_df for group_key, group_df in groups_same_day_quantity_price_cusip_tradetype_flags.items() if set(group_df['trade_type']) == {'S'} or set(group_df['trade_type']) == {'P'}}
    if verbose: print(f'Number of groups: {len(groups_same_day_quantity_price_cusip_tradetype_flags_nonDD)}')

    all_indices_to_remove = []
    for group_df in groups_same_day_quantity_price_cusip_tradetype_flags_nonDD.values():
        df, row_indices_to_remove = _combine_trades(df, group_df)
        all_indices_to_remove.extend(row_indices_to_remove)
    df = df.drop(all_indices_to_remove)
    if verbose: print(f'Number of trades: {len(df)}')

    groups_same_day_quantity_price_cusip_tradetype_flags_DD = {group_key: group_df for group_key, group_df in groups_same_day_quantity_price_cusip_tradetype_flags.items() if set(group_df['trade_type']) == {'D'}}
    if verbose: print(f'Number of groups: {len(groups_same_day_quantity_price_cusip_tradetype_flags_DD)}')

    all_indices_to_remove = []
    for group_df in groups_same_day_quantity_price_cusip_tradetype_flags_DD.values():
        _, indices_to_remove = _get_most_recent_index_and_others(group_df, True)
        all_indices_to_remove.extend(indices_to_remove)
    df = df.drop(all_indices_to_remove)
    if verbose: print(f'Number of trades: {len(df)}')

    groups_same_day_quantity_price_cusip = df_select_features.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'quantity', 'dollar_price', 'cusip'])
    groups_same_day_quantity_price_cusip = {group_key: group_df for group_key, group_df in groups_same_day_quantity_price_cusip if len(group_df) > 1}    # removes groups with only 1 item since this is not really a group
    if verbose: print(f'Number of groups: {len(groups_same_day_quantity_price_cusip)}')

    two_trades_only_one_is_dd = lambda df: len(df) == 2 and 'D' in set(df['trade_type']) and len(set(df['trade_type'])) == 2    # checks whether `df` has two trades where only one is an inter-dealer trade
    groups_same_day_quantity_price_cusip_dd_2 = {group_key: group_df for group_key, group_df in groups_same_day_quantity_price_cusip.items() if two_trades_only_one_is_dd(group_df)}
    if verbose: print(f'Number of groups: {len(groups_same_day_quantity_price_cusip_dd_2)}')

    all_indices_to_remove = []
    for group_df in groups_same_day_quantity_price_cusip_dd_2.values():
        interdealer_trade_index = group_df[group_df['trade_type'] == 'D'].index[0]
        other_trade = group_df[group_df['trade_type'] != 'D']
        other_trade_index = other_trade.index[0]
        if other_trade['is_non_transaction_based_compensation'].values[0]:    # .values[0] isolates the value for this trade
            all_indices_to_remove.append(other_trade_index)
        else:
            all_indices_to_remove.append(interdealer_trade_index)
    df = df.drop(all_indices_to_remove)
    
    if verbose:
        print(f'Number of trades in original dataframe: {len(df_original)}')
        print(f'Number of trades after combining and removing trades: {len(df)}')
        print(f'Number of trades removed: {len(df_original) - len(df)}')

    return df