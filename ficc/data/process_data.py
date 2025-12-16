'''
'''
import warnings
import numpy as np

from ficc.utils.process_features import process_features
from ficc.utils.initialize_pandarallel import initialize_pandarallel

from ficc.data.process_trade_history import process_trade_history
from ficc.utils.auxiliary_functions import convert_dates
from ficc.utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE
from ficc.utils.get_treasury_rate import get_treasury_rate_dict
from ficc.utils.adding_flags import add_bookkeeping_flag, add_replica_count_flag, add_same_day_flag, add_ntbc_precursor_flag
from ficc.utils.related_trade import add_related_trades
from ficc.utils.yield_curve_params import yield_curve_params


def process_data(query, 
                 bq_client, 
                 num_trades_in_history: int, 
                 num_features_for_each_trade_in_history: int, 
                 file_path: str, 
                 yield_curve='FICC_NEW', 
                 remove_short_maturity: bool = False, 
                 trade_history_delay: int = 12, 
                 min_trades_in_history: int = 0, 
                 use_treasury_spread: bool = False, 
                 add_flags: bool = False, 
                 add_related_trades_bool: bool = False, 
                 add_rtrs_in_history: bool = False, 
                 only_dollar_price_history: bool = False, 
                 save_data: bool = True, 
                 process_similar_trades_history: bool = False, 
                 use_multiprocessing: bool = True, 
                 end_of_day: bool = False, 
                 performing_automated_training: bool = False,
                 **kwargs):
    if len(kwargs) != 0: warnings.warn(f'**kwargs is not empty and has following arguments: {kwargs.keys()}', category=RuntimeWarning)
        
    yield_curve = yield_curve.upper()
    if yield_curve == 'FICC' or yield_curve == 'FICC_NEW':
        print(f'Grabbing {"end of day" if end_of_day else "real-time (minute)"} yield curve params')
        try:
            nelson_params, scalar_params, shape_parameter = yield_curve_params(bq_client, yield_curve, end_of_day)
        except Exception as e:
            print('Unable to grab yield curve parameters')
            raise e
    
    arguments_to_print_for_calling_process_trade_history = ['remove_short_maturity', 'trade_history_delay', 'use_treasury_spread', 'min_trades_in_history', 'add_flags', 'add_related_trades_bool', 'add_rtrs_in_history', 'only_dollar_price_history', 'save_data', 'process_similar_trades_history', 'end_of_day']
    locals_args_dict = locals()    # must create this dictionary to avoid using `locals()` in the list comprehension which otherwise causes an error due to scoping
    joined_arguments_and_values = "\n\t".join([f"{arg}: {locals_args_dict[arg]}" for arg in arguments_to_print_for_calling_process_trade_history])
    print(f'Calling `process_trade_history(...)` with\n\t{joined_arguments_and_values}')
    treasury_rate_dict = get_treasury_rate_dict(bq_client) if use_treasury_spread is True else None
    trades_df = process_trade_history(query, 
                                      bq_client, 
                                      num_trades_in_history, 
                                      num_features_for_each_trade_in_history, 
                                      file_path, 
                                      remove_short_maturity, 
                                      trade_history_delay,  
                                      min_trades_in_history, 
                                      use_treasury_spread, 
                                      add_rtrs_in_history, 
                                      only_dollar_price_history, 
                                      yield_curve, 
                                      treasury_rate_dict, 
                                      nelson_params, 
                                      scalar_params, 
                                      shape_parameter, 
                                      save_data, 
                                      process_similar_trades_history, 
                                      end_of_day=end_of_day, 
                                      performing_automated_training=performing_automated_training)
    if use_multiprocessing: initialize_pandarallel()
    
    if trades_df is None: return None    # no new trades

    if len(trades_df) == 0:
        print(f'After dropping trades for not having a treasury rate, the dataframe is empty')
        return None
        
    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)
    trades_df = process_features(trades_df)

    if remove_short_maturity is True:
        print('Removing short maturity')
        trades_df = trades_df[trades_df.days_to_maturity >= np.log10(1 + 400)]

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)

    if add_flags is True:    # add additional flags to the data
        # trades_df = add_replica_flag(trades_df)    # the IS_REPLICA flag was originally designed to remove replica trades from the trade history
        trades_df = add_replica_count_flag(trades_df)
        trades_df = add_bookkeeping_flag(trades_df)
        trades_df = add_same_day_flag(trades_df)
        trades_df = add_ntbc_precursor_flag(trades_df)
    
    if add_related_trades_bool is True:
        print('Adding most recent related trade')
        trades_df = add_related_trades(trades_df,
                                       RELATED_TRADE_FEATURE_PREFIX, 
                                       NUM_RELATED_TRADES, 
                                       CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE)
    
    print(f'{len(trades_df)} trades at the end of `process_data(...)` ranging from trade datetimes of {trades_df.trade_datetime.min()} to {trades_df.trade_datetime.max()}')
    return trades_df
