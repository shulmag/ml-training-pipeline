'''
'''
import os
import pickle

import pandas as pd

from google.cloud import storage

from ficc.utils.auxiliary_functions import sqltodf, process_ratings, check_if_pickle_file_exists_and_matches_query
from ficc.utils.pad_trade_history import pad_trade_history
from ficc.utils.trade_list_to_array import trade_list_to_array
from ficc.utils.initialize_pandarallel import initialize_pandarallel
from ficc.utils.gcp_storage_functions import upload_data


def fetch_trade_data(query: str, bq_client, path: str = 'data.pkl', save_data: bool = True, performing_automated_training: bool = False):
    trades_df = check_if_pickle_file_exists_and_matches_query(query, path)
    if trades_df is not None:
        print(f'Using cached data from {path} since the query matches the one in the file')
        return trades_df
    
    print(f'Grabbing data from BigQuery with query:')
    print(query)
    trades_df = sqltodf(query, bq_client)

    if save_data:
        print(f'Saving query and data to {path}')
        dir_name = os.path.dirname(path)
        if dir_name:    # Only make directories if there's a directory in the path, i.e., `dir_name` is not '' or `None`
            os.makedirs(dir_name, exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
        with open(path, 'wb') as f: 
            pickle.dump((query, trades_df), f)

        if performing_automated_training:
            print(f'Also saving data to {path} in Google Cloud Storage')
            upload_data(storage.Client(), 'automated_training', f'raw_data/{path}', path)
    return trades_df


def restrict_number_of_trades(trade_history_series: pd.Series, num_trades: int, processing_similar_trades: bool) -> pd.Series:
    '''`processing_similar_trades` is used solely for print output.'''
    trade_history_prefix = 'similar ' if processing_similar_trades else ''
    print(f'Restricting the {trade_history_prefix}trade history to the {num_trades} most recent trades')
    return trade_history_series.apply(lambda history: history[:num_trades])


def pad_trade_history_column(series: pd.Series, num_trades_in_history: int, min_trades_in_history: int, num_features_for_each_trade_in_history: int, processing_similar_trades: bool, use_multiprocessing: bool = True) -> pd.Series:
    '''`processing_similar_trades` is used solely for print output.'''
    if use_multiprocessing: initialize_pandarallel()
    trade_history_prefix = 'similar ' if processing_similar_trades else ''
    print(f'Padding {trade_history_prefix}trade history')
    print(f'Minimum number of trades required in the {trade_history_prefix}trade history: {min_trades_in_history}')
    apply_func = series.parallel_apply if use_multiprocessing else series.apply
    return apply_func(pad_trade_history, args=[num_trades_in_history, num_features_for_each_trade_in_history, min_trades_in_history])


def restrict_number_of_trades_and_pad_trade_history(df: pd.DataFrame, trade_history_column_name: str, num_trades_in_history: int, min_trades_in_history: int, num_features_for_each_trade_in_history: int, processing_similar_trades: bool = False, use_multiprocessing: bool = True) -> pd.DataFrame:
    '''`processing_similar_trades` is used solely for print output.'''
    df[trade_history_column_name] = restrict_number_of_trades(df[trade_history_column_name], num_trades_in_history, processing_similar_trades)
    df[trade_history_column_name] = pad_trade_history_column(df[trade_history_column_name], num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history, processing_similar_trades, use_multiprocessing)
    return df


def process_trade_history(query: str,
                          bq_client, 
                          num_trades_in_history: int, 
                          num_features_for_each_trade_in_history: int, 
                          file_path: str,  
                          remove_short_maturity: bool,
                          trade_history_delay: int, 
                          min_trades_in_history: int, 
                          use_treasury_spread: bool,
                          add_rtrs_in_history: bool,
                          only_dollar_price_history: bool, 
                          yield_curve_to_use: str, 
                          treasury_rate_dict: dict, 
                          nelson_params: dict, 
                          scalar_params: dict, 
                          shape_parameter: dict, 
                          save_data: bool = True, 
                          process_similar_trades_history: bool = False, 
                          use_multiprocessing: bool = True, 
                          end_of_day: bool = False, 
                          performing_automated_training: bool = False) -> pd.DataFrame:
    if use_multiprocessing: initialize_pandarallel()
    trades_df = fetch_trade_data(query, bq_client, file_path, save_data, performing_automated_training)
    if len(trades_df) == 0:
        print('Raw data contains 0 trades')
        return None
    print(f'Raw data contains {len(trades_df)} trades ranging from trade datetimes of {trades_df.trade_datetime.min()} to {trades_df.trade_datetime.max()}')
    
    trades_df = process_ratings(trades_df)
    # trades_df = convert_object_to_category(trades_df)

    print('Creating trade history')
    if remove_short_maturity is True: print('Removing trades with shorter maturity')
    print(f'Removing trades less than {trade_history_delay} seconds in the history')
    
    processed_trade_history_column_name = 'trade_history'
    last_features_column_name = 'temp_last_features'
    processed_trades_df = pd.DataFrame(data=None, index=trades_df.index, columns=[processed_trade_history_column_name, last_features_column_name])
    unprocessed_trade_history_column_name = 'recent'
    apply_func = trades_df[unprocessed_trade_history_column_name].parallel_apply if use_multiprocessing else trades_df[unprocessed_trade_history_column_name].apply
    processed_trades_df = apply_func(trade_list_to_array, args=([remove_short_maturity,
                                                                 trade_history_delay,
                                                                 use_treasury_spread,
                                                                 add_rtrs_in_history,
                                                                 only_dollar_price_history, 
                                                                 yield_curve_to_use, 
                                                                 treasury_rate_dict, 
                                                                 nelson_params, 
                                                                 scalar_params, 
                                                                 shape_parameter, 
                                                                 end_of_day]))
                                                                        
    trades_df[[processed_trade_history_column_name, last_features_column_name]] = pd.DataFrame(processed_trades_df.tolist(), index=trades_df.index)
    del processed_trades_df
    print('Trade history created')
    print('Getting last trade features')
    trades_df[['last_yield_spread', 
               'last_ficc_ycl', 
               'last_rtrs_control_number', 
               'last_yield', 
               'last_dollar_price', 
               'last_seconds_ago', 
               'last_size', 
               'last_calc_date', 
               'last_maturity_date', 
               'last_next_call_date', 
               'last_par_call_date', 
               'last_refund_date', 
               'last_trade_datetime', 
               'last_calc_day_cat', 
               'last_settlement_date', 
               'last_trade_type']] = pd.DataFrame(trades_df[last_features_column_name].tolist(), index=trades_df.index)
    trades_df = trades_df.drop(columns=[last_features_column_name, unprocessed_trade_history_column_name])
    trades_df = restrict_number_of_trades_and_pad_trade_history(trades_df, processed_trade_history_column_name, num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history, use_multiprocessing=use_multiprocessing)
    trade_history_features = [processed_trade_history_column_name]

    if process_similar_trades_history is True:
        print('Creating similar trade history')
        processed_trade_history_column_name = 'similar_trade_history'
        last_features_column_name = 'temp_last_similar_features'
        processed_trades_df = pd.DataFrame(data=None, index=trades_df.index, columns=[processed_trade_history_column_name, last_features_column_name])
        unprocessed_trade_history_column_name = 'recent_5_year_mat'
        apply_func = trades_df[unprocessed_trade_history_column_name].parallel_apply if use_multiprocessing else trades_df[unprocessed_trade_history_column_name].apply
        processed_trades_df = apply_func(trade_list_to_array, args=([remove_short_maturity,
                                                                     trade_history_delay,
                                                                     use_treasury_spread,
                                                                     add_rtrs_in_history,
                                                                     only_dollar_price_history, 
                                                                     yield_curve_to_use, 
                                                                     treasury_rate_dict, 
                                                                     nelson_params, 
                                                                     scalar_params, 
                                                                     shape_parameter, 
                                                                     end_of_day]))
        # TODO: speed the below line up by not storing the unnecessary information for the most recent trade (needed when processing same CUSIP trade history, but not for similar trade history)
        trades_df[[processed_trade_history_column_name, last_features_column_name]] = pd.DataFrame(processed_trades_df.tolist(), index=trades_df.index)
        del processed_trades_df
        print('Similar trade history created')
        trades_df = trades_df.drop(columns=[last_features_column_name, unprocessed_trade_history_column_name])
        trades_df = restrict_number_of_trades_and_pad_trade_history(trades_df, processed_trade_history_column_name, num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history, True, use_multiprocessing=use_multiprocessing)
        trade_history_features.append(processed_trade_history_column_name)
    
    num_trades_before_removing_null_history = len(trades_df)
    trades_df.dropna(subset=trade_history_features, inplace=True)
    print(f'Processed trade history contains {len(trades_df)} trades. Prior to removing null histories (i.e., removing null values in {trade_history_features}), it contained {num_trades_before_removing_null_history} trades.')
    return trades_df
