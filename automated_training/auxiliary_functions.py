'''
'''
import warnings
import math
import subprocess
import traceback    # used to print out the stack trace when there is an error
import os
import sys
import shutil
import holidays
import pickle

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from sklearn import preprocessing
from datetime import datetime

from google.cloud import bigquery
from google.cloud import storage
from google.api_core.exceptions import NotFound as GCPNotFoundException

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from automated_training.auxiliary_variables import CATEGORICAL_FEATURES, \
                                                    CATEGORICAL_FEATURES_DOLLAR_PRICE, \
                                                    NON_CAT_FEATURES, \
                                                    NON_CAT_FEATURES_DOLLAR_PRICE, \
                                                    BINARY, \
                                                    BINARY_DOLLAR_PRICE, \
                                                    PREDICTORS, \
                                                    PREDICTORS_DOLLAR_PRICE, \
                                                    YS_VARIANTS, \
                                                    YS_FEATS, \
                                                    DP_VARIANTS, \
                                                    DP_FEATS, \
                                                    YEAR_MONTH_DAY, \
                                                    HOUR_MIN_SEC, \
                                                    QUERY_FEATURES, \
                                                    QUERY_CONDITIONS, \
                                                    ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL, \
                                                    ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL, \
                                                    ADDITIONAL_QUERY_FEATURES_FOR_YIELD_SPREAD_WITH_SIMILAR_TRADES_MODEL, \
                                                    EASTERN, \
                                                    BUSINESS_DAY, \
                                                    NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL, \
                                                    NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL, \
                                                    CATEGORICAL_FEATURES_VALUES, \
                                                    SAVE_MODEL_AND_DATA, \
                                                    HOME_DIRECTORY, \
                                                    WORKING_DIRECTORY, \
                                                    PROJECT_ID, \
                                                    AUXILIARY_VIEWS_DATASET_NAME, \
                                                    BUCKET_NAME, \
                                                    TRAINING_LOGS_DIRECTORY, \
                                                    MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK, \
                                                    EARLIEST_TRADE_DATETIME, \
                                                    MAX_NUM_DAYS_IN_THE_PAST_TO_KEEP_DATA, \
                                                    MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, \
                                                    OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD, \
                                                    OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE, \
                                                    TTYPE_DICT, \
                                                    LONG_TIME_AGO_IN_NUM_SECONDS, \
                                                    MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY, \
                                                    HISTORICAL_PREDICTION_TABLE, \
                                                    EMAIL_RECIPIENTS, \
                                                    SENDER_EMAIL, \
                                                    BATCH_SIZE, \
                                                    NUM_EPOCHS, \
                                                    MODEL_NAME_TO_ARCHIVED_MODEL_FOLDER, \
                                                    TESTING, \
                                                    USE_PICKLED_DATA, \
                                                    ROW_NAME_DETERMINING_MODEL_SWITCH, \
                                                    USE_END_OF_DAY_YIELD_CURVE_COEFFICIENTS
from automated_training.yield_with_similar_trades_model import yield_spread_with_similar_trades_model
from automated_training.dollar_model import dollar_price_model
from automated_training.set_random_seed import set_seed

from ficc.utils.gcp_storage_functions import upload_data, download_data
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_functions import function_timer, get_ys_trade_history_features, get_dp_trade_history_features
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.initialize_pandarallel import initialize_pandarallel
from ficc.utils.get_treasury_rate import get_treasury_rate_dict, current_treasury_rate
from ficc.utils.yc_data import add_yield_curve


set_seed()


# this variable needs to be in this file instead of `auxiliary_variables.py` since importing the models in `auxiliary_variables.py` causes a circular import error
MODEL_NAME_TO_KERAS_MODEL = {'dollar_price': dollar_price_model, 
                             'yield_spread_with_similar_trades': yield_spread_with_similar_trades_model}


def get_creds():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gil/git/ficc/creds.json'
    return None


def get_storage_client():
    get_creds()
    return storage.Client()


def get_bq_client():
    get_creds()
    return bigquery.Client()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()


def setup_gpus(install_nvidia_drivers_if_gpu_not_present: bool = True, force_cpu: bool = False):
    import tensorflow as tf    # lazy loading for lower latency

    if force_cpu:
        print(f'Forcing use of CPU instead of GPU')
        tf.config.set_visible_devices([], 'GPU')
        return
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        warnings.warn('No GPUs found')
        if install_nvidia_drivers_if_gpu_not_present:
            shell_script_path = f'{WORKING_DIRECTORY}/install-cuda_12_2_0-linux-x86_64-debian-11-network.sh'
            if os.path.isfile(shell_script_path):
                shell_script_output = subprocess.check_output(['sh', shell_script_path])
                print(f'Output of running shell script from {shell_script_path}:\n{shell_script_output.decode()}')
            else:
                print(f'{shell_script_path} not found')
            setup_gpus(False)    # if GPU still not found after installing CUDA toolkit, then proceed with training without the GPU
    else:
        print(f'Found {len(gpus)} GPUs: {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)    # used so that multiple models can be trained on the same GPU since TensorFlow by default allocates the entire GPU RAM for a single training instance; https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory


D_prev = dict()
P_prev = dict()
S_prev = dict()


def check_that_model_is_supported(model: str):
    '''Raises an AssertionError if `model` is not supported.'''
    supported_models = ['yield_spread', 'yield_spread_with_similar_trades', 'dollar_price']
    assert model in supported_models, f'Model should be {" or ".join(supported_models)}, but was instead: {model}'
    return model


def get_trade_history_columns(model: str) -> list:
    '''Creates a list of columns.'''
    check_that_model_is_supported(model)
    if 'yield_spread' in model:
        variants, features = YS_VARIANTS, YS_FEATS
    else:
        variants, features = DP_VARIANTS, DP_FEATS
    
    columns = []
    for prefix in variants:
        for suffix in features:
            columns.append(prefix + suffix)
    return columns


def target_trade_processing_for_attention(row):
    trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
    return np.tile(target_trade_features, (1, 1))


@function_timer
def add_treasury_spread(data: pd.DataFrame, use_multiprocessing: bool = True) -> pd.DataFrame:
    assert 'new_ficc_ycl' in data.columns, '`new_ficc_ycl` column is not present in the data. Please call `add_yield_curve(...)` before calling this function.'
    treasury_rate_dict = get_treasury_rate_dict(BQ_CLIENT)
    columns_needed_for_treasury_rate = ['trade_date', 'calc_date', 'settlement_date', 'maturity_date']
    treasury_rate_apply_func = data[columns_needed_for_treasury_rate].parallel_apply if use_multiprocessing else data[columns_needed_for_treasury_rate].apply
    data['treasury_rate'] = treasury_rate_apply_func(lambda trade: current_treasury_rate(treasury_rate_dict, trade), axis=1)
    null_treasury_rate = data['treasury_rate'].isnull()
    if null_treasury_rate.sum() > 0:
        trade_dates_corresponding_to_null_treasury_rate = data.loc[null_treasury_rate, 'trade_date']
        print(f'The following `trade_date`s have no corresponding `treasury_rate`, so all {null_treasury_rate.sum()} trades with these `trade_date`s have been removed: {trade_dates_corresponding_to_null_treasury_rate.unique()}')
        data = data[~null_treasury_rate]
    data['ficc_treasury_spread'] = data['new_ficc_ycl'] - (data['treasury_rate'] * 100)
    return data


def decrement_week_days(date: str, num_week_days: int) -> str:
    '''Subtract `num_week_days` from `date`. Using `BusinessDay` instead of `CustomBusinessDay` with the `USFederalHolidayCalendar` since 
    we do not want to skip holidays when using archived models since the desired model may have been created on a holiday, which is fine 
    because that model was trained with data before the holiday.'''
    return (datetime.strptime(date, YEAR_MONTH_DAY) - BusinessDay(num_week_days)).strftime(YEAR_MONTH_DAY)


def increment_week_days(date: str, num_week_days: int) -> str:
    '''Add `num_week_days` to `date`. Using `BusinessDay` instead of `CustomBusinessDay` with the `USFederalHolidayCalendar` since 
    we do not want to skip holidays when checking whether a future date is a holiday.'''
    return (datetime.strptime(date, YEAR_MONTH_DAY) + BusinessDay(num_week_days)).strftime(YEAR_MONTH_DAY)


def decrement_business_days(date: str, num_business_days: int) -> str:
    '''Subtract `num_business_days` from `date`.'''
    return (datetime.strptime(date, YEAR_MONTH_DAY) - (BUSINESS_DAY * num_business_days)).strftime(YEAR_MONTH_DAY)


def increment_business_days(date: str, num_business_days: int) -> str:
    '''Subtract `num_business_days` from `date`.'''
    return (datetime.strptime(date, YEAR_MONTH_DAY) + (BUSINESS_DAY * num_business_days)).strftime(YEAR_MONTH_DAY)


def is_a_holiday(date: str) -> bool:
    '''Determine whether `date` is a US national holiday.'''
    date = datetime.strptime(date, YEAR_MONTH_DAY)
    holidays_US = holidays.US()
    if date in holidays_US:
        print(f'{date} is a national holiday so we do not expect there to be new trades on this day')
        return True
    return False


def earliest_trade_from_new_data_is_same_as_last_trade_date(new_data: pd.DataFrame, last_trade_date) -> bool:
    '''Checks whether `last_trade_date` is the same as the date of the earliest trade in `new_data`. This 
    situation arises materialized trade history is created in the middle of the day, and so there are trades 
    on the same day that are still coming in. If we do not account for this case, then the automated training 
    fails since it searches for trades to populate the testing set as those after the `last_trade_date`.'''
    return new_data.trade_date.min().date().strftime(YEAR_MONTH_DAY) == last_trade_date


def check_no_duplicate_rtrs_control_numbers(data: pd.DataFrame) -> None:
    '''Raise an AssertionError if there are duplicate RTRS control numbers in `data`.
    
    >>> try:
    ...     check_no_duplicate_rtrs_control_numbers(pd.DataFrame({'rtrs_control_number': [101, 102, 103, 101, 104, 102, 105, 103]}))
    ... except AssertionError as _:
    ...     print('Successfully raised an AssertionError')
    Successfully raised an AssertionError
    >>> check_no_duplicate_rtrs_control_numbers(pd.DataFrame({'rtrs_control_number': [101, 102, 103]}))
    '''
    rtrs_control_numbers = data['rtrs_control_number']
    duplicate_rtrs_control_numbers = rtrs_control_numbers[rtrs_control_numbers.duplicated()].to_numpy()
    num_duplicate_rtrs_control_numbers = len(duplicate_rtrs_control_numbers)
    assert num_duplicate_rtrs_control_numbers == 0, f'There are {num_duplicate_rtrs_control_numbers} duplicate RTRS control numbers. Here are the first 10:\n\t{duplicate_rtrs_control_numbers[:10]}'


@function_timer
def get_new_data(file_name, 
                 model: str, 
                 use_treasury_spread: bool = False, 
                 optional_arguments_for_process_data: dict = dict(), 
                 data_query: str = None, 
                 save_data: bool = SAVE_MODEL_AND_DATA, 
                 use_multiprocessing: bool = True, 
                 raw_data_file_path: str = None, 
                 performing_automated_training: bool = False) -> tuple:
    '''`data_query` will always be `None` unless the user is attempting to get processed data for a specific 
    slice of data by calling `get_new_data(...)` from another function. If `performing_automated_training` is `True`, 
    then the raw data file will be uploaded to Google Cloud Storage.'''
    check_that_model_is_supported(model)
    old_data, last_trade_datetime, last_trade_date = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
    print(f'last trade datetime: {last_trade_datetime}')
    if data_query is None: data_query = get_data_query(last_trade_datetime, model)
    file_date = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY)

    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if 'yield_spread' in model else get_dp_trade_history_features()
    num_features_for_each_trade_in_history = len(trade_history_features)
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if 'yield_spread' in model else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    if raw_data_file_path is None: raw_data_file_path = f'raw_data_{file_date}_{model}.pkl'
    data_from_last_trade_datetime = process_data(data_query, 
                                                 BQ_CLIENT, 
                                                 num_trades_in_history, 
                                                 num_features_for_each_trade_in_history, 
                                                 raw_data_file_path, 
                                                 save_data=save_data, 
                                                 process_similar_trades_history=(model == 'yield_spread_with_similar_trades'), 
                                                 use_multiprocessing=use_multiprocessing, 
                                                 performing_automated_training=performing_automated_training, 
                                                 end_of_day=USE_END_OF_DAY_YIELD_CURVE_COEFFICIENTS, 
                                                 **optional_arguments_for_process_data)
    
    if data_from_last_trade_datetime is not None:
        check_no_duplicate_rtrs_control_numbers(data_from_last_trade_datetime)
        if earliest_trade_from_new_data_is_same_as_last_trade_date(data_from_last_trade_datetime, last_trade_date):    # see explanation in docstring for `earliest_trade_from_new_data_is_same_as_last_trade_date(...)` as to why this scenario is important to handle
            decremented_last_trade_date = decrement_business_days(last_trade_date, 1)
            warnings.warn(f'Since the earliest trade from the new data is the same as the last trade date, we are decrementing the last trade date from {last_trade_date} to {decremented_last_trade_date}. This occurs because materialized trade history was created in the middle of the work day. If materialized trade history was not created during the middle of the work day, then investigate why we are inside this `if` statement.')
            last_trade_date = decremented_last_trade_date
        
        if model == 'dollar_price': data_from_last_trade_datetime = data_from_last_trade_datetime.rename(columns={'trade_history': 'trade_history_dollar_price'})    # change the trade history column name to match with `PREDICTORS_DOLLAR_PRICE`
    return old_data, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_file_path


def remove_old_trades(data: pd.DataFrame, num_days_to_keep: int, most_recent_trade_date: str = None, dataset_name: str = None) -> pd.DataFrame:
    '''Only keep `num_days_to_keep` days from the most recent trade in `data`. `dataset_name` is used only for print output.'''
    from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
    most_recent_trade_date = data.trade_date.max() if most_recent_trade_date is None else pd.to_datetime(most_recent_trade_date)
    days_to_most_recent_trade = diff_in_days_two_dates(most_recent_trade_date, data.trade_date, 'exact')
    print(f'Removing trades{from_dataset_name} older than {num_days_to_keep} days before {most_recent_trade_date}')
    return data[days_to_most_recent_trade < num_days_to_keep]


@function_timer
def combine_new_data_with_old_data(old_data: pd.DataFrame, new_data: pd.DataFrame, model: str, use_treasury_spread: bool) -> pd.DataFrame:
    initialize_pandarallel()    # only initialize if needed
    check_that_model_is_supported(model)
    if new_data is None: return old_data    # there is new data since `last_trade_date`

    num_trades_in_new_data = len(new_data)
    num_trades_in_old_data = 0 if old_data is None else len(old_data)
    print(f'Old data has {num_trades_in_old_data} trades. New data has {num_trades_in_new_data} trades')

    trade_history_feature_names = ['trade_history'] if 'yield_spread' in model else ['trade_history_dollar_price']
    if model == 'yield_spread_with_similar_trades': trade_history_feature_names.append('similar_trade_history')
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if 'yield_spread' in model else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    
    print(f'Restricting history to {num_trades_in_history} trades')
    for trade_history_feature_name in trade_history_feature_names:
        new_data[trade_history_feature_name] = new_data[trade_history_feature_name].apply(lambda history: history[:num_trades_in_history])    # this line is redundant because this procedure is done in `process_data(...)`, but keeping it for now in case that functionality changes to no longer truncate upstream
        if old_data is not None:
            old_data[trade_history_feature_name] = old_data[trade_history_feature_name].apply(lambda history: history[:num_trades_in_history])    # done in case `num_trades_in_history` has decreased from before

    new_data['yield'] = new_data['yield'] * 100
    if 'yield_spread' in model:
        new_data = add_yield_curve(new_data, BQ_CLIENT, end_of_day=USE_END_OF_DAY_YIELD_CURVE_COEFFICIENTS)    # adds `new_ficc_ycl` column to `new_data`
        if use_treasury_spread: new_data = add_treasury_spread(new_data)    # adds `ficc_treasury_spread` column to `new_data`
    new_data['target_attention_features'] = new_data.parallel_apply(target_trade_processing_for_attention, axis=1)

    trade_history_sum_features = []    # remove these features after checking for null values
    for trade_history_feature_name in trade_history_feature_names:
        # the below 5 lines are redundant because this procedure is done in `process_data(...)`, but keeping it for now in case that functionality changes to no longer truncate upstream
        trade_history_sum_feature_name = f'{trade_history_feature_name}_sum'
        trade_history_sum_features.append(trade_history_sum_feature_name)
        new_data[trade_history_sum_feature_name] = new_data[trade_history_feature_name].parallel_apply(lambda history: np.sum(history))
        new_data.dropna(inplace=True, subset=[trade_history_sum_feature_name])
        print(f'Removed {num_trades_in_new_data - len(new_data)} trades, since these have null values in the trade history')
    
    new_data.issue_amount = new_data.issue_amount.replace([np.inf, -np.inf], np.nan)

    data = pd.concat([new_data, old_data], sort=False) if old_data is not None else new_data    # concatenating `new_data` to the original `data` dataframe; `sort=False` is used to keep the original order of the columns
    data = data.drop(columns=trade_history_sum_features)
    if 'yield_spread' in model: data['new_ys'] = data['yield'] - data['new_ficc_ycl']
    print(f'{len(data)} trades after combining new and old data')
    return data


@function_timer
def add_trade_history_derived_features(data: pd.DataFrame, model: str, use_treasury_spread: bool = False) -> pd.DataFrame:
    initialize_pandarallel()    # only initialize if needed
    check_that_model_is_supported(model)
    data = data.sort_values('trade_datetime', ascending=True)    # when calling `trade_history_derived_features...(...)` the order of trades needs to be ascending for `trade_datetime`
    trade_history_derived_features = trade_history_derived_features_yield_spread(use_treasury_spread) if 'yield_spread' in model else trade_history_derived_features_dollar_price
    trade_history_feature_name = 'trade_history' if 'yield_spread' in model else 'trade_history_dollar_price'
    
    temp = data[['cusip', trade_history_feature_name, 'quantity', 'trade_type']].parallel_apply(trade_history_derived_features, axis=1)
    cols = get_trade_history_columns(model)
    data[cols] = pd.DataFrame(temp.tolist(), index=data.index)
    del temp

    data = data.sort_values('trade_datetime', ascending=False)    # reset the order of the data to `trade_datetime` descending which is what is assumed to be the original order
    return data


@function_timer
def drop_features_with_null_value(df: pd.DataFrame, model: str) -> pd.DataFrame:
    check_that_model_is_supported(model)
    predictors = PREDICTORS if 'yield_spread' in model else PREDICTORS_DOLLAR_PRICE
    if model == 'yield_spread_with_similar_trades': predictors.append('similar_trade_history')
    # df = df.dropna(subset=features)
    for feature in predictors:    # perform the procedure feature by feature to output how many trades are being removed for each feature
        num_trades_before = len(df)
        df = df.dropna(subset=[feature])
        num_trades_after = len(df)
        if num_trades_before != num_trades_after: print(f'Removed {num_trades_before - num_trades_after} trades for having a null value in feature: {feature}')
    return df


@function_timer
def save_data(data: pd.DataFrame, file_name: str, upload_to_google_cloud_bucket: bool = True) -> None:
    file_path = f'{WORKING_DIRECTORY}/files/{file_name}'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
    data = remove_old_trades(data, MAX_NUM_DAYS_IN_THE_PAST_TO_KEEP_DATA, dataset_name='entire processed data file')
    print(f'Saving data to pickle file with name {file_path}')
    data.to_pickle(file_path)
    if upload_to_google_cloud_bucket:
        folder_prefix = 'processed_data/' if file_name.startswith('processed_data_') else ''
        upload_data(STORAGE_CLIENT, BUCKET_NAME, f'{folder_prefix}{file_name}', file_path)


def _get_trade_date_where_data_exists(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int, exclusions_function: callable, on_or_after: str = 'after'):
    '''Iterate backwards on `date` until the data after `date` is non-empty. Go back a maximum of 
    `max_number_of_business_days_to_go_back` days. If `exclusions_function` is not `None`, assumes 
    that the function returns values, where the first item is the data after exclusions, and the 
    second item is the data before exclusions.'''
    valid_values_for_on_or_after = ('on', 'after')
    assert on_or_after in valid_values_for_on_or_after, f'Invalid value for `on_or_after`: {on_or_after}. Must be one of {valid_values_for_on_or_after}'

    def get_data_on_or_after_date(date_of_interest: str) -> pd.DataFrame:
        if on_or_after == 'after':
            data_for_date = data[data.trade_date > date_of_interest]
            data_for_date_earliest_date = data_for_date.trade_date.min()
            data_for_date = data_for_date[data_for_date.trade_date == data_for_date_earliest_date]    # restrict `data_for_date` to have only one day of trades
        else:
            data_for_date = data[data.trade_date == date_of_interest]
        return data_for_date
    
    previous_date = date
    data_for_date = get_data_on_or_after_date(previous_date)
    if exclusions_function is not None: data_for_date, _ = exclusions_function(data_for_date)
    business_days_gone_back = 0
    while len(data_for_date) < MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY and business_days_gone_back < max_number_of_business_days_to_go_back:
        business_days_gone_back += 1
        previous_date = decrement_business_days(date, business_days_gone_back)
        data_for_date = get_data_on_or_after_date(previous_date)
        if exclusions_function is not None: data_for_date, _ = exclusions_function(data_for_date)
    if business_days_gone_back == max_number_of_business_days_to_go_back:
        print(f'Went back {business_days_gone_back} and could not find any data; not going back any further, so returning the original `date`')
        return date
    return previous_date


def get_trade_date_where_data_exists_after_this_date(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int = 10, exclusions_function: callable = None):
    if not TESTING: return date
    return _get_trade_date_where_data_exists(date, data, max_number_of_business_days_to_go_back, exclusions_function, 'after')


def get_trade_date_where_data_exists_on_this_date(date, data: pd.DataFrame, max_number_of_business_days_to_go_back: int = 10, exclusions_function: callable = None):
    return _get_trade_date_where_data_exists(date, data, max_number_of_business_days_to_go_back, exclusions_function, 'on')


def get_feature_as_array(df: pd.DataFrame, feature_name: str) -> np.array:
    '''Extract `feature_name` from `df` and return a numpy representation that can be used by the model.'''
    return np.stack(df[feature_name].to_numpy())


@function_timer
def create_input(data: pd.DataFrame, encoders: dict, model: str, ignore_label: bool = False):
    check_that_model_is_supported(model)

    datalist = []
    if model == 'yield_spread_with_similar_trades': datalist.append(get_feature_as_array(data, 'similar_trade_history'))
    trade_history_feature_name = 'trade_history' if 'yield_spread' in model else 'trade_history_dollar_price'
    datalist.append(get_feature_as_array(data, trade_history_feature_name))
    datalist.append(get_feature_as_array(data, 'target_attention_features'))

    categorical_features = CATEGORICAL_FEATURES if 'yield_spread' in model else CATEGORICAL_FEATURES_DOLLAR_PRICE
    non_cat_features = NON_CAT_FEATURES if 'yield_spread' in model else NON_CAT_FEATURES_DOLLAR_PRICE
    binary_features = BINARY if 'yield_spread' in model else BINARY_DOLLAR_PRICE

    noncat_and_binary = []
    for feature in non_cat_features + binary_features:
        noncat_and_binary.append(np.expand_dims(data[feature].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for feature in categorical_features:
        encoded = encoders[feature].transform(data[feature])
        datalist.append(encoded.astype('float32'))

    if ignore_label:
        labels = None
    else:
        label_name = 'new_ys' if 'yield_spread' in model else 'dollar_price'
        labels = data[label_name]
        assert all(data_input.shape[0] == labels.shape[0] for data_input in datalist), f'Mismatch between datalist inputs and labels: {[data_input.shape[0] for data_input in datalist]} vs {labels.shape[0]}'
    return datalist, labels


def get_data_and_last_trade_datetime(bucket_name: str, file_name: str):
    '''Get the dataframe from `bucket_name/file_name` and the most recent trade datetime from this dataframe.'''
    data = None if file_name is None else download_data(STORAGE_CLIENT, bucket_name, f'processed_data/{file_name}')
    if data is None: return None, EARLIEST_TRADE_DATETIME, EARLIEST_TRADE_DATETIME[:10]    # get trades starting from `EARLIEST_TRADE_DATETIME` if we do not have these trades already in a pickle file; string representation of datetime has the date as the first 10 characters (YYYY-MM-DD is 10 characters)
    last_trade_datetime = data.trade_datetime.max().strftime(YEAR_MONTH_DAY + 'T' + HOUR_MIN_SEC)
    last_trade_date = data.trade_date.max().date().strftime(YEAR_MONTH_DAY)
    return data, last_trade_datetime, last_trade_date


def is_string_representation_of_a_date(date_string: str, date_format: str = YEAR_MONTH_DAY) -> bool:
    '''Check if a string can be represented as a date in the given format.

    >>> is_string_representation_of_a_date('2024-11-18')
    True
    >>> is_string_representation_of_a_date('2024-02-29')
    True
    >>> is_string_representation_of_a_date('2023-02-29')
    False
    >>> is_string_representation_of_a_date('18/11/2024', '%d/%m/%Y')
    True
    >>> is_string_representation_of_a_date('11-18-2024', '%m-%d-%Y')
    True
    >>> is_string_representation_of_a_date('2024-13-01')
    False
    >>> is_string_representation_of_a_date('not-a-date')
    False
    >>> is_string_representation_of_a_date('', '%Y-%m-%d')
    False
    '''
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False


def get_data_query(last_trade_datetime: str,    # may be a string representation of a date instead of a datetime
                   model: str, 
                   latest_trade_date_to_query: str = None) -> str:    # may be a string representation of a datetime instead of a date
    '''`latest_trade_date_to_query` is a string representation of either a date or a datetime.'''
    check_that_model_is_supported(model)
    if last_trade_datetime == EARLIEST_TRADE_DATETIME: raise RuntimeError('Use `ficc_python/get_processed_data.py to get the data for this query since it will most likely crash due to memory issues if done in this function')
    query_features = QUERY_FEATURES
    query_conditions = QUERY_CONDITIONS
    if 'yield_spread' in model:
        query_conditions = ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL + query_conditions
    else:
        query_features = query_features + ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL

    if model == 'yield_spread_with_similar_trades': query_features = ADDITIONAL_QUERY_FEATURES_FOR_YIELD_SPREAD_WITH_SIMILAR_TRADES_MODEL + query_features

    features_as_string = ', '.join(query_features)
    if is_string_representation_of_a_date(last_trade_datetime): last_trade_datetime = f'{last_trade_datetime}T00:00:00'
    query_conditions = query_conditions + [f'trade_datetime > "{last_trade_datetime}"']
    if latest_trade_date_to_query is not None:
        if is_string_representation_of_a_date(latest_trade_date_to_query): latest_trade_date_to_query = f'{latest_trade_date_to_query}T23:59:59'
        query_conditions = query_conditions + [f'trade_datetime < "{latest_trade_date_to_query}"']
    conditions_as_string = ' AND '.join(query_conditions)
    return f'''SELECT {features_as_string}
               FROM `{PROJECT_ID}.{AUXILIARY_VIEWS_DATASET_NAME}.trade_history_same_issue_5_yr_mat_bucket_1_materialized`
               WHERE {conditions_as_string}
               ORDER BY trade_datetime DESC'''


def get_optional_arguments_for_process_data(model):
    check_that_model_is_supported(model)
    return OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD if 'yield_spread' in model else OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE


def update_data(model: str, 
                performing_automated_training: bool = False) -> tuple:
    check_that_model_is_supported(model)
    filename = MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[model]
    optional_arguments_for_process_data = get_optional_arguments_for_process_data(model)
    use_treasury_spread = optional_arguments_for_process_data.get('use_treasury_spread', False)
    data_before_last_trade_datetime, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = get_new_data(filename, 
                                                                                                                                                              model, 
                                                                                                                                                              use_treasury_spread=use_treasury_spread, 
                                                                                                                                                              performing_automated_training=performing_automated_training,
                                                                                                                                                              optional_arguments_for_process_data=optional_arguments_for_process_data)
    if data_from_last_trade_datetime is not None:    # no need to continue this procedure if there are no new trades since the below subprocedures were performed on the data before storing it on Google Cloud Storage
        data = combine_new_data_with_old_data(data_before_last_trade_datetime, data_from_last_trade_datetime, model, use_treasury_spread)
        data = add_trade_history_derived_features(data, model, use_treasury_spread)
        data = drop_features_with_null_value(data, model)
        if SAVE_MODEL_AND_DATA: save_data(data, filename)
    else:
        data = data_before_last_trade_datetime
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


@function_timer
def save_update_data_results_to_pickle_files(model: str, 
                                             performing_automated_training: bool = False):
    '''The function specified in `update_data` is called, and the 3 return values are stored as pickle files. If 
    testing, then first check whether the pickle files exist, before calling `update_data`. `suffix` is appended 
    to the end of the filename for each pickle file.'''
    check_that_model_is_supported(model)
    data_pickle_filepath = f'{WORKING_DIRECTORY}/files/data_from_update_data_{model}.pkl'
    last_trade_data_from_update_data_pickle_filepath = f'{WORKING_DIRECTORY}/files/last_trade_data_from_update_data_{model}.pkl'
    num_features_for_each_trade_in_history_pickle_filepath = f'{WORKING_DIRECTORY}/files/num_features_for_each_trade_in_history_{model}.pkl'

    os.makedirs(f'{WORKING_DIRECTORY}/files', exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
    if USE_PICKLED_DATA and os.path.isfile(data_pickle_filepath):
        print(f'Found a data file in {data_pickle_filepath}, so no need to run update_data(...)')
        raw_data_filepath = None
        data = pd.read_pickle(data_pickle_filepath)
        with open(last_trade_data_from_update_data_pickle_filepath, 'rb') as file: last_trade_date = pickle.load(file)
        with open(num_features_for_each_trade_in_history_pickle_filepath, 'rb') as file: num_features_for_each_trade_in_history = pickle.load(file)
    else:
        data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = update_data(model, performing_automated_training)
        data.to_pickle(data_pickle_filepath)
        with open(last_trade_data_from_update_data_pickle_filepath, 'wb') as file: pickle.dump(last_trade_date, file)
        with open(num_features_for_each_trade_in_history_pickle_filepath, 'wb') as file: pickle.dump(num_features_for_each_trade_in_history, file)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def fit_encoders(data: pd.DataFrame, categorical_features: list, model: str):
    '''Fits label encoders to categorical features in the data. For a few of the categorical features, the values 
    don't change for these features we use the pre-defined set of values specified in `CATEGORICAL_FEATURES_VALUES`. 
    Outputs a tuple of dictionaries where the first item is the encoders and the second item is the maximum value 
    for each class.'''
    check_that_model_is_supported(model)
    encoders = {}
    fmax = {}
    for feature in categorical_features:
        if feature in CATEGORICAL_FEATURES_VALUES:
            fprep = preprocessing.LabelEncoder().fit(CATEGORICAL_FEATURES_VALUES[feature])
        else:
            fprep = preprocessing.LabelEncoder().fit(data[feature].drop_duplicates())
        fmax[feature] = np.max(fprep.transform(fprep.classes_))
        encoders[feature] = fprep
    
    encoders_filename = get_encoders_filename(model)
    if os.path.exists(WORKING_DIRECTORY):
        with open(f'{WORKING_DIRECTORY}/{encoders_filename}', 'wb') as file:
            pickle.dump(encoders, file)
    else:
        print(f'{WORKING_DIRECTORY} does not exist, so {WORKING_DIRECTORY}/{encoders_filename} was not written to')
    return encoders, fmax


def _trade_history_derived_features(row, model: str, use_treasury_spread: bool = False) -> list:
    check_that_model_is_supported(model)
    if model == 'yield_spread':
        variants = YS_VARIANTS
        trade_history_features = get_ys_trade_history_features(use_treasury_spread)
    else:
        variants = DP_VARIANTS
        trade_history_features = get_dp_trade_history_features()

    label_feature = 'yield_spread' if 'yield_spread' in model else 'dollar_price'
    ys_or_dp_idx = trade_history_features.index(label_feature)
    par_traded_idx = trade_history_features.index('par_traded')
    trade_type1_idx = trade_history_features.index('trade_type1')
    trade_type2_idx = trade_history_features.index('trade_type2')
    seconds_ago_idx = trade_history_features.index('seconds_ago')


    def extract_feature_from_trade(row, name, trade):    # `name` is used solely for debugging
        ys_or_dp = trade[ys_or_dp_idx]
        ttypes = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])] + row.trade_type
        seconds_ago = trade[seconds_ago_idx]
        quantity_diff = np.log10(1 + np.abs(10**trade[par_traded_idx] - 10**row.quantity))
        return [ys_or_dp, ttypes, seconds_ago, quantity_diff]


    global D_prev
    global S_prev
    global P_prev
    
    trade_history_feature_name = 'trade_history' if 'yield_spread' in model else 'trade_history_dollar_price'
    trade_history = row[trade_history_feature_name]
    most_recent_trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip, most_recent_trade)
    D_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS        

    P_min_ago_t = P_prev.get(row.cusip, most_recent_trade)
    P_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    S_min_ago_t = S_prev.get(row.cusip, most_recent_trade)
    S_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    max_ys_or_dp_t = most_recent_trade
    max_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    min_ys_or_dp_t = most_recent_trade
    min_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    max_qty_t = most_recent_trade
    max_qty = most_recent_trade[par_traded_idx]
    min_ago_t = most_recent_trade
    min_ago = most_recent_trade[seconds_ago_idx]
    
    for trade in trade_history:
        seconds_ago = trade[seconds_ago_idx]
        # Checking if the first trade in the history is from the same block; TODO: shouldn't this be checked for every trade?
        if seconds_ago == 0: continue

        ys_or_dp = trade[ys_or_dp_idx]
        if ys_or_dp > max_ys_or_dp: 
            max_ys_or_dp_t = trade
            max_ys_or_dp =ys_or_dp
        elif ys_or_dp < min_ys_or_dp: 
            min_ys_or_dp_t = trade
            min_ys_or_dp = ys_or_dp

        par_traded = trade[par_traded_idx]
        if par_traded > max_qty: 
            max_qty_t = trade 
            max_qty = par_traded

        if seconds_ago < min_ago:    # TODO: isn't this just the most recent trade not in the same block, and isn't this initialized above already?
            min_ago_t = trade
            min_ago = seconds_ago
            
        side = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])]
        if side == 'D':
            if seconds_ago < D_min_ago: 
                D_min_ago_t = trade
                D_min_ago = seconds_ago
                D_prev[row.cusip] = trade
        elif side == 'P':
            if seconds_ago < P_min_ago: 
                P_min_ago_t = trade
                P_min_ago = seconds_ago
                P_prev[row.cusip] = trade
        elif side == 'S':
            if seconds_ago < S_min_ago: 
                S_min_ago_t = trade
                S_min_ago = seconds_ago
                S_prev[row.cusip] = trade
        else: 
            print('invalid side', trade)
    
    variant_trade_dict = dict(zip(variants, [max_ys_or_dp_t, min_ys_or_dp_t, max_qty_t, min_ago_t, D_min_ago_t, P_min_ago_t, S_min_ago_t]))
    variant_trade_list = []
    for variant_name, variant_trade in variant_trade_dict.items():
        feature_list = extract_feature_from_trade(row, variant_name, variant_trade)
        variant_trade_list += feature_list
    return variant_trade_list


def trade_history_derived_features_yield_spread(use_treasury_spread) -> callable:
    return lambda row: _trade_history_derived_features(row, 'yield_spread', use_treasury_spread)
def trade_history_derived_features_dollar_price(row) -> list:
    return _trade_history_derived_features(row, 'dollar_price')


def get_early_stopping_callbacks(loss_type_to_monitor: str, patience: int):
    from tensorflow.keras.callbacks import EarlyStopping    # lazy loading for lower latency

    LOSS_TYPES_TO_MONITOR = {'validation', 'training'}
    assert loss_type_to_monitor in LOSS_TYPES_TO_MONITOR, f'`loss_type_to_monitor` must be one of {LOSS_TYPES_TO_MONITOR} but was instead {loss_type_to_monitor}'
    assert patience > 0, f'`patience`: {patience} must be greater than 0'
    LOSS_MAPPING = {'validation': 'val_loss', 'training': 'loss'}
    return [EarlyStopping(monitor=LOSS_MAPPING[loss_type_to_monitor], 
                          patience=patience, 
                          verbose=1,  
                          restore_best_weights=True)]


def combine_two_histories(history1, history2):
    from tensorflow.keras.callbacks import History    # lazy loading for lower latency

    combined_history_dict = {}
    all_keys = set(history1.history.keys()).union(history2.history.keys())    # Combine all unique keys
    for key in all_keys:
        # Fill missing values with NaN for the history that lacks the key
        history1_values = history1.history.get(key, [np.nan] * len(history2.history.get(key, [])))
        history2_values = history2.history.get(key, [np.nan] * len(history1.history.get(key, [])))
        combined_history_dict[key] = history1_values + history2_values
    
    # Create a new History object and stored `combined_history_dict`
    combined_history = History()
    combined_history.history = combined_history_dict
    return combined_history


def get_train_and_validation_set(inputs: list, labels: np.ndarray, validation_split: float):
    '''Assumes that the most recent data is at the end. `inputs` is a datalist and so each item in the list is an input of size `labels.shape[0]`.'''
    assert 0 <= validation_split < 1, f'`validation_split`: {validation_split} must be in [0, 1)'
    num_data_points_for_validation = math.ceil(validation_split * labels.shape[0])
    x_val = [x_input[-num_data_points_for_validation:] for x_input in inputs]    # `inputs` is a datalist and so each item in the list is an input of size `labels.shape[0]`
    y_val = labels[-num_data_points_for_validation:]
    x_train = [x_input[:-num_data_points_for_validation] for x_input in inputs]    # `inputs` is a datalist and so each item in the list is an input of size `labels.shape[0]`
    y_train = labels[:-num_data_points_for_validation]
    return x_train, y_train, x_val, y_val


def fit_model(model, inputs: list, labels: np.ndarray, epochs: int, loss_type_to_monitor: str = None, patience: int = None, **kwargs):
    if loss_type_to_monitor is None or patience is None:
        print(f'Not using any callbacks because one of `loss_type_to_monitor`: {loss_type_to_monitor} and `patience`: {patience} is `None`')
        callbacks = None
    else:
        print(f'Using early stopping callback for {loss_type_to_monitor} loss with patience of {patience} epochs')
        callbacks = get_early_stopping_callbacks(loss_type_to_monitor, patience)
    return model.fit(inputs, 
                     labels, 
                     epochs=epochs, 
                     batch_size=BATCH_SIZE, 
                     verbose=1,    # prints out the progress bar; set to 2 to just have one line per epoch
                     callbacks=callbacks, 
                     use_multiprocessing=True, 
                     workers=8, 
                     shuffle=False,    # setting `shuffle=False` since it was more accurate during experiments; originally thought to set `shuffle=True` since shuffling data for each epoch would lead to better generalization; shuffling is okay here because the input data is for separate CUSIPs and so the training does not need to maintain the original order of the data and does not learn anything temporal between instances
                     **kwargs)


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, optimizer: str = 'Adam'):
    '''Two-phase training. Phase 1: train on the entire dataset leaving some of the data to be the validation set. Phase 2: 
    train on only the validation set for a small number of epochs. Phase 1 learns large patterns in the data while focusing 
    on validation loss to ensure generalization. Phase 2 trains only on the validation set to allow these datapoints to 
    influence the weights so the model has exposure to the most recent data, but is done for a small number of epochs since 
    there is no validation set to ensure generalization. Assumes that `*_train` is in ascending order of time (`trade_datetime`). 
    Past experiments to choose a good training procedure: https://ficcai.atlassian.net/browse/FA-2461.'''
    from tensorflow import keras    # lazy loading for lower latency
    
    # this variable needs to be in this file instead of `auxiliary_variables.py` since initializing tensorflow in another file causes `setup_gpus(...)` to fail
    # NOTE: SGD does not work on Apple Metal GPU
    SUPPORTED_OPTIMIZERS = {'Adam': keras.optimizers.Adam(learning_rate=0.0001), 
                            'SGD': keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9)}    # 0.9 is a well-tested industry / academic default for `momentum`
    
    assert optimizer in SUPPORTED_OPTIMIZERS, f'optimizer: {optimizer} must be in {SUPPORTED_OPTIMIZERS.keys()}'
    model.compile(optimizer=SUPPORTED_OPTIMIZERS[optimizer], 
                  loss=keras.losses.MeanAbsoluteError(), 
                  metrics=[keras.metrics.MeanAbsoluteError()])

    validation_split = 0.1    # fraction of the data to be used as validation data
    x_train, y_train, x_val, y_val = get_train_and_validation_set(x_train, y_train, validation_split)

    # phase 1: train on the entire dataset leaving some of the data to be the validation set
    patience = math.ceil(0.2 * NUM_EPOCHS)    # generally recommended patience from ChatGPT
    history_optimizing_val_loss = fit_model(model, 
                                            x_train, 
                                            y_train, 
                                            NUM_EPOCHS, 
                                            'validation', 
                                            patience, 
                                            validation_data=(x_val, y_val))
    # phase 2: train on only the validation set for a small number of epochs
    history_optimizing_training_loss = fit_model(model, 
                                                 x_val, 
                                                 y_val, 
                                                 math.ceil(NUM_EPOCHS * validation_split), 
                                                 'training', 
                                                 math.ceil(patience * validation_split))
    # evaluate on test set
    _, mae = model.evaluate(x_test, 
                            y_test, 
                            verbose=1, 
                            batch_size=BATCH_SIZE)
    return model, mae, combine_two_histories(history_optimizing_val_loss, history_optimizing_training_loss)


def segment_results(data: pd.DataFrame, absolute_difference: np.array) -> pd.DataFrame:
    '''Return a dataframe that has the MAE and count for specific slices of `data`.'''
    def get_mae_and_count(condition=None):
        if condition is not None:
            delta_cond, data_cond = absolute_difference[condition], data[condition]
        else:
            delta_cond, data_cond = absolute_difference, data
        return np.round(np.mean(delta_cond), 3), data_cond.shape[0]    # round mae to 3 digits after the decimal point to reduce noise

    inter_dealer = data.trade_type == 'D'
    dealer_purchase = data.trade_type == 'P'
    dealer_sell = data.trade_type == 'S'
    aaa = data.rating == 'AAA'
    investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    investment_grade = data.rating.isin(investment_grade_ratings)
    par_traded_greater_than_or_equal_to_100k = data.par_traded >= 1e5
    
    last_days_ago = data.last_seconds_ago / (60 * 60 * 24)
    last_trade_date_within_a_week = last_days_ago <= 7
    last_trade_date_week_to_two_weeks = (7 < last_days_ago) & (last_days_ago <= 14)
    last_trade_date_two_weeks_to_four_weeks = (14 < last_days_ago) & (last_days_ago <= 28)
    last_trade_date_more_than_four_weeks = 28 < last_days_ago

    result_df = pd.DataFrame(data=[get_mae_and_count(),
                                   get_mae_and_count(inter_dealer),
                                   get_mae_and_count(dealer_purchase),
                                   get_mae_and_count(dealer_sell), 
                                   get_mae_and_count(aaa), 
                                   get_mae_and_count(investment_grade),
                                   get_mae_and_count(par_traded_greater_than_or_equal_to_100k), 
                                   get_mae_and_count(last_trade_date_within_a_week), 
                                   get_mae_and_count(last_trade_date_week_to_two_weeks), 
                                   get_mae_and_count(last_trade_date_two_weeks_to_four_weeks), 
                                   get_mae_and_count(last_trade_date_more_than_four_weeks)],
                             columns=['Mean Absolute Error', 'Trade Count'],
                             index=['Entire set', 
                                    'Dealer-Dealer', 
                                    'Bid Side / Dealer-Purchase', 
                                    'Offered Side / Dealer-Sell', 
                                    'AAA', 
                                    'Investment Grade', 
                                    'Trade size >= 100k', 
                                    'Last trade <= 7 days', 
                                    '7 days < Last trade <= 14 days', 
                                    '14 days < Last trade <= 28 days', 
                                    '28 days < Last trade'])
    return result_df


def get_table_schema_for_predictions():
    '''Returns the schema required for the bigquery table storing the predictions.'''
    schema = [
        bigquery.SchemaField('rtrs_control_number', 'INTEGER', 'REQUIRED'),
        bigquery.SchemaField('cusip', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('trade_date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('dollar_price', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('yield','FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ficc_ycl', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys_prediction', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('prediction_datetime', 'DATETIME', 'REQUIRED')
    ] 
    return schema


def upload_predictions(data: pd.DataFrame, model: str) -> str:
    '''Upload predictions to BigQuery. Returns the table name.'''
    assert model in HISTORICAL_PREDICTION_TABLE, f'Trying to upload predictions for {model}, but can only upload predictions for the following models: {list(HISTORICAL_PREDICTION_TABLE.keys())}'
    table_name = HISTORICAL_PREDICTION_TABLE[model]

    # check if the table exists, and if not, then create it
    try:
        BQ_CLIENT.get_table(table_name)
    except GCPNotFoundException:
        print(f'Table {table_name} does not exist. Creating it...')
        schema = get_table_schema_for_predictions()
        table = bigquery.Table(table_name, schema=schema)
        BQ_CLIENT.create_table(table)

    job_config = bigquery.LoadJobConfig(schema=get_table_schema_for_predictions(), write_disposition='WRITE_APPEND')
    job = BQ_CLIENT.load_table_from_dataframe(data, table_name, job_config=job_config)
    try:
        job.result()
        print(f'Upload successful to {table_name}')
    except Exception as e:
        print(f'Failed to upload to {table_name} with {type(e)}: {e}')
        raise e


def load_model_from_date(date: str, folder: str, bucket: str):
    '''Taken almost directly from `point_in_time_pricing_timestamp.py`.
    When using the `cache_output` decorator, we should not have any optional arguments as this may interfere with 
    how the cache lookup is done (optional arguments may not be put into the args set).
    As of 2024-06-07, we assume that the model name has the entire YYYY-MM-DD in the name.'''
    from tensorflow import keras    # lazy loading for lower latency

    archived_folder_to_model_name = {archived_folder: model_name for model_name, archived_folder in MODEL_NAME_TO_ARCHIVED_MODEL_FOLDER.items()}
    model = archived_folder_to_model_name[folder]
    check_that_model_is_supported(model)
    
    # `model_prefix` should match the naming convention of MODEL_NAME in the associated .sh script
    if model == 'yield_spread':
        model_prefix = ''
    elif model == 'dollar_price':
        model_prefix = 'dollar-v2-'
    else:    # model == yield_spread_with_similar_trades
        model_prefix = 'similar-trades-v2-'

    bucket_folder_model_path = os.path.join(os.path.join(bucket, folder), f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<folder>/<model>
    base_model_path = os.path.join(bucket, f'{model_prefix}model-{date}')    # create path of the form: <bucket>/<model>
    for model_path in (bucket_folder_model_path, base_model_path):    # iterate over possible paths and try to load the model
        print(f'Attempting to load model from {model_path}')
        try:
            keras_model = keras.models.load_model(model_path)
            print(f'Model loaded from {model_path}')
            return keras_model
        except Exception as e:
            print(f'Model failed to load from {model_path} with exception: {e}')


@function_timer
def load_model(date_of_interest: str, model: str, max_num_week_days_in_the_past_to_check: int = MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK, bucket: str = 'gs://'+BUCKET_NAME):
    '''Taken almost directly from `point_in_time_pricing_timestamp.py`.
    This function finds the appropriate model, either in the automated_training directory, or in a special directory. 
    TODO: clean up the way we store models on cloud storage by unifying the folders and naming convention.'''
    folder = MODEL_NAME_TO_ARCHIVED_MODEL_FOLDER[model]
    for num_business_days_in_the_past in range(max_num_week_days_in_the_past_to_check):
        model_date_string = decrement_week_days(date_of_interest, num_business_days_in_the_past)    # do not want to skip holidays because the desired model may have been created on a holiday, which is fine because that model was trained with data before the holiday
        loaded_model = load_model_from_date(model_date_string, folder, bucket)
        if loaded_model is not None: return loaded_model, model_date_string
    raise FileNotFoundError(f'No model for {folder} was found from {date_of_interest} to {model_date_string}')
    

def create_summary_of_results(model, data: pd.DataFrame, inputs: list, labels: list, print_results: bool = True, return_predictions_and_delta: bool = False):
    '''Creates a dataframe that can be sent as a table over email for the performance of `model` on 
    `inputs` validated on `labels`. `inputs` and `labels` are transformed using `create_input(...)` 
    from `data` to be able to be used by the `model` for inference.'''
    result_df = pd.DataFrame()
    predictions, delta = None, None
    try:
        predictions = model.predict(inputs, batch_size=BATCH_SIZE).flatten()
        delta = np.abs(predictions - labels)
        result_df = segment_results(data, delta)
    except Exception as e:
        print(f'Unable to create results dataframe with `segment_results(...)`. {type(e)}:', e)
        print('Stack trace:')
        print(traceback.format_exc())

    if print_results:
        try:
            print(result_df.to_markdown())
        except Exception as e:
            print(f'Unable to display results dataframe with .to_markdown(). Need to run `pip install tabulate` on this machine in order to display the dataframe in an easy to read way. {type(e)}:', e)
    return (result_df, predictions, delta) if return_predictions_and_delta else result_df


def not_enough_trades_in_test_data(test_data, min_trades_to_use_test_data):
    '''Return `None` for the number of arguments that `train_model(...)` returns.'''
    print(f'No model is trained since there are only {len(test_data)} trades in `test_data`, which is less than our chosen threshold of {min_trades_to_use_test_data}; `train_model(...)` is terminated')
    return None, None, None, None, None, None, None, ''


@function_timer
def train_model(data: pd.DataFrame, last_trade_date: str, model: str, num_features_for_each_trade_in_history: int, date_for_previous_model: str = None, exclusions_function: callable = None):
    '''The final return value is a string that should be in the beginning of the email that is sent out 
    which provides transparency on the training procedure. If `date_for_previous_model` is `None`, then 
    do not attempt to load a previous model for accuracy comparison.'''
    check_that_model_is_supported(model)
    train_model_text_list = []    # store important data from the training procedure that will be outputted in summary email
    data = remove_old_trades(data, 240, last_trade_date, dataset_name='training/testing dataset')    # 240 = 8 * 30, so we are using approximately 8 months of data for training
    categorical_features = CATEGORICAL_FEATURES if 'yield_spread' in model else CATEGORICAL_FEATURES_DOLLAR_PRICE
    encoders, fmax = fit_encoders(data, categorical_features, model)
    
    if TESTING:
        last_trade_date = data.trade_date.max().strftime(YEAR_MONTH_DAY)
        last_trade_date = get_trade_date_where_data_exists_after_this_date(last_trade_date, data, exclusions_function=exclusions_function)
    test_data = data[data.trade_date > last_trade_date]    # `test_data` can only contain trades after `last_trade_date`
    test_data_date = test_data.trade_date.min()
    test_data = test_data[test_data.trade_date == test_data_date]    # restrict `test_data` to have only one day of trades
    if len(test_data) < MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY:
        return not_enough_trades_in_test_data(test_data, MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY)
    test_data_date = test_data_date.strftime(YEAR_MONTH_DAY)
    if exclusions_function is not None:
        test_data, test_data_before_exclusions = exclusions_function(test_data, 'test_data')
    else:
        test_data_before_exclusions = test_data
    
    train_data = data[data.trade_date <= last_trade_date]    # `train_data` only contains trades before and including `last_trade_date`
    training_set_info = f'Training set contains {len(train_data)} trades ranging from trade datetimes of {train_data.trade_datetime.min()} to {train_data.trade_datetime.max()}'
    test_set_info = f'Test set contains {len(test_data)} trades ranging from trade datetimes of {test_data.trade_datetime.min()} to {test_data.trade_datetime.max()}'
    print(training_set_info)
    print(test_set_info)
    train_model_text_list.extend([training_set_info, test_set_info])

    non_cat_features = NON_CAT_FEATURES if 'yield_spread' in model else NON_CAT_FEATURES_DOLLAR_PRICE
    binary = BINARY if 'yield_spread' in model else BINARY_DOLLAR_PRICE

    column_to_be_sorted_by = 'trade_datetime'
    if not train_data[column_to_be_sorted_by].is_monotonic_increasing:
        print(f'Sorting the data by {column_to_be_sorted_by} ascending to be able to separate the training data and the validation data by {column_to_be_sorted_by}')
        train_data = train_data.sort_values(column_to_be_sorted_by, ascending=True)    # sort by `column_to_be_sorted_by` so further downstream operations (e.g., creating the validation set) is done with respect to the time series nature of the data

    x_train, y_train = create_input(train_data, encoders, model)
    x_test, y_test = create_input(test_data, encoders, model)

    keras_model = MODEL_NAME_TO_KERAS_MODEL[model]
    num_trades_in_history = NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL if 'yield_spread' in model else NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL
    untrained_model = keras_model(x_train, 
                                  num_trades_in_history, 
                                  num_features_for_each_trade_in_history, 
                                  categorical_features, 
                                  non_cat_features, 
                                  binary, 
                                  fmax)
    trained_model, mae, history = train_and_evaluate_model(untrained_model, x_train, y_train, x_test, y_test)

    create_summary_of_results_for_test_data = lambda model: create_summary_of_results(model, test_data, x_test, y_test)
    result_df = create_summary_of_results_for_test_data(trained_model)
    if date_for_previous_model is None:
        previous_business_date_model, previous_business_date_model_date, result_df_using_previous_day_model = None, None, None
    else:
        try:
            previous_business_date_model, previous_business_date_model_date = load_model(date_for_previous_model, model)
            result_df_using_previous_day_model = create_summary_of_results_for_test_data(previous_business_date_model)
        except Exception as e:
            print(f'Unable to create the dataframe for the model evaluation email due to {type(e)}: {e}')
            print('Stack trace:')
            print(traceback.format_exc())
            result_df_using_previous_day_model = None
            if 'previous_business_date_model' not in locals(): previous_business_date_model, previous_business_date_model_date = None, None
    
    # uploading predictions to bigquery (only for yield spread model)
    if SAVE_MODEL_AND_DATA and 'yield_spread' in model:
        try:
            test_data_before_exclusions_x_test, test_data_before_exclusions_y_test = create_input(test_data_before_exclusions, encoders, model)

            print('Creating summary of results for test data before exclusions ...')
            create_summary_of_results(trained_model, test_data_before_exclusions, test_data_before_exclusions_x_test, test_data_before_exclusions_y_test)
            
            test_data_before_exclusions['new_ys_prediction'] = trained_model.predict(test_data_before_exclusions_x_test, batch_size=BATCH_SIZE)
            test_data_before_exclusions = test_data_before_exclusions[['rtrs_control_number', 'cusip', 'trade_date', 'dollar_price', 'yield', 'new_ficc_ycl', 'new_ys', 'new_ys_prediction']]
            test_data_before_exclusions['prediction_datetime'] = pd.to_datetime(datetime.now(EASTERN).replace(microsecond=0))
            test_data_before_exclusions['trade_date'] = pd.to_datetime(test_data_before_exclusions['trade_date']).dt.date
            
            upload_predictions(test_data_before_exclusions, model)
        except Exception as e:
            print(f'Failed to upload predictions to BigQuery. {type(e)}:', e)
    return trained_model, test_data_date, previous_business_date_model, previous_business_date_model_date, encoders, mae, (result_df, result_df_using_previous_day_model), '<br>'.join(train_model_text_list)    # use '<br>' for the separator since this will create a new line in the HTML body that will be sent out by email


@function_timer
def get_model_results(data: pd.DataFrame, trade_date: str, model: str, loaded_model, encoders: dict, exclusions_function: callable = None) -> pd.DataFrame:
    '''NOTE: `model` is a string that denotes whether we are working with the yield spread model or 
    the dollar price model, and `loaded_model` is an actual keras model. This may cause confusion.
    If `exclusions_function` is not `None`, assumes that the function returns values, where the first 
    item is the data after exclusions, and the second item is the data before exclusions.'''
    check_that_model_is_supported(model)
    data_on_trade_date = data[data.trade_date == trade_date]
    if exclusions_function is not None: data_on_trade_date, _ = exclusions_function(data_on_trade_date)
    inputs, labels = create_input(data_on_trade_date, encoders, model)
    return create_summary_of_results(loaded_model, data_on_trade_date, inputs, labels)


def _get_model_suffix(model: str, wo_underscore: bool = False):
    '''Return the model suffix for creating filenames to store encoders and trained models for `model`. 
    This function is used as a subprocedure in `get_encoders_filename(...)` and `get_model_zip_filename(...). 
    `wo_underscore` is an optional boolean argument that removes the first character of the suffix which 
    is expected to be the underscore.'''
    check_that_model_is_supported(model)
    if model == 'yield_spread':
        suffix = ''
    elif model == 'yield_spread_with_similar_trades':
        suffix = '_similar_trades'
    else:
        suffix = '_dollar_price'
    if wo_underscore:
        suffix = suffix[1:] if suffix != '' else suffix    # assumes that the underscore to remove is the first character
    return suffix


def get_encoders_filename(model: str):
    '''Return the filename of the pickle file that stores the encoders for `model`.'''
    suffix = _get_model_suffix(model)
    return f'encoders{suffix}.pkl'


def get_model_zip_filename(model: str):
    '''Return the filename of the zip file that stores the trained model for `model`.'''
    suffix = _get_model_suffix(model)
    return f'model{suffix}_v2'


@function_timer
def save_model(trained_model, 
               encoders, 
               model: str, 
               model_file_path: str = None,    # used to override the default `model_file_path` defined in this function for trained models in production
               upload_to_google_cloud_bucket: bool = True):
    '''NOTE: `model` is a string that denotes whether we are working with the yield spread with similar trades model 
    or the dollar price model, and `trained_model` is an actual keras model, which may cause confusion.'''
    check_that_model_is_supported(model)
    if trained_model is None:
        print('trained_model is `None` and so not saving it to storage')
        return None
    
    suffix_wo_underscore = _get_model_suffix(model, True)    # need `suffix_wo_underscore` variable as well since past implementations of this function have model naming missing an underscore
    file_timestamp = datetime.now(EASTERN).strftime(YEAR_MONTH_DAY + '-%H-%M')
    print(f'file time stamp: {file_timestamp}')

    if encoders is not None:
        encoders_filename = get_encoders_filename(model)
        encoders_directory = f'{WORKING_DIRECTORY}/files'
        encoders_filepath = f'{encoders_directory}/{encoders_filename}'
        print(f'Saving encoders to {encoders_filepath}')
        os.makedirs(encoders_directory, exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
        with open(encoders_filepath, 'wb') as file:
            pickle.dump(encoders, file)    
        if upload_to_google_cloud_bucket: upload_data(STORAGE_CLIENT, BUCKET_NAME, encoders_filename, encoders_filepath)

    if model_file_path is None:
        folder = f'{model}_models'
        saved_models_directory = f'{HOME_DIRECTORY}/trained_models/{folder}/saved_models'
        os.makedirs(saved_models_directory, exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
        model_file_path = f'{saved_models_directory}/saved_model_{suffix_wo_underscore}{file_timestamp}'
    print(f'Saving model to {model_file_path}')
    trained_model.save(model_file_path)
    
    if upload_to_google_cloud_bucket: 
        model_zip_filename = get_model_zip_filename(model)
        model_zip_filepath = f'{HOME_DIRECTORY}/trained_models/{model_zip_filename}'
        shutil.make_archive(model_zip_filepath, 'zip', model_file_path)
        
        upload_data(STORAGE_CLIENT, BUCKET_NAME, f'{model_zip_filename}.zip', f'{model_zip_filepath}.zip')
        os.system(f'rm -r {model_file_path}')


def remove_file(file_path: str) -> None:
    '''Remove the file at path: `file_path`. 
    Taken directly from ChatGPT from search: "how to remove a file python".'''
    try:
        os.remove(file_path)
        print(f"File '{file_path}' removed successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"PermissionError: Unable to remove '{file_path}'.")
    except Exception as e:
        print(f'{type(e)}: {e}')


def train_save_evaluate_model(model: str, exclusions_function: callable = None, current_date: str = None, performing_automated_training: bool = False) -> bool:
    '''Returns a boolean indicating whether the model traffic should be switched based to the newly trained model.'''
    check_that_model_is_supported(model)
    print(f'Python version: {sys.version}')
    current_datetime = datetime.now(EASTERN)
    print(f'automated_training_{model}_model.py starting at {current_datetime} ET')

    data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = save_update_data_results_to_pickle_files(model, performing_automated_training)
    if data is None or len(data) == 0: raise RuntimeError('`data` is empty')
    if current_date is None:
        current_date = current_datetime.date().strftime(YEAR_MONTH_DAY)
        previous_business_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(current_date, 1), data)    # ensures that the business day has trades on it
    else:
        print(f'Using the argument when calling the script as the current date: {current_date}')
        previous_business_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(current_date, 1), data)    # ensures that the business day has trades on it
        last_trade_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(previous_business_date, 1), data)    # ensures that the business day has trades on it

    current_date_model, test_data_date, previous_business_date_model, previous_business_date_model_date, encoders, mae, mae_df_list, email_intro_text = train_model(data, last_trade_date, model, num_features_for_each_trade_in_history, previous_business_date, exclusions_function)
    if mae_df_list is not None:
        current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df = mae_df_list
        try:
            business_date_before_test_data_date = get_trade_date_where_data_exists_on_this_date(decrement_business_days(test_data_date, 1), data)    # ensures that the business day has trades on it
            assert previous_business_date_model is not None, f'Raising an AssertionError since previous_business_date_model is `None`, which will run the cleanup logic in the `except` clause'
            business_date_before_test_data_date_data_previous_business_date_model_result_df = get_model_results(data, business_date_before_test_data_date, model, previous_business_date_model, encoders, exclusions_function)
        except Exception as e:
            print(f'Unable to create the third dataframe used in the model evaluation email due to {type(e)}: {e}')
            print('Stack trace:')
            print(traceback.format_exc())
            business_date_before_test_data_date = None
            business_date_before_test_data_date_data_previous_business_date_model_result_df = None

    if raw_data_filepath is not None:
        print(f'Removing {raw_data_filepath} since {model} training is complete')
        remove_file(raw_data_filepath)

    if not TESTING and current_date_model is None:
        send_no_new_model_email(last_trade_date, EMAIL_RECIPIENTS, model)
        raise RuntimeError(f'No new data was found for {model} training, so the procedure is terminating gracefully and without issue. Raising an error only so that the shell script terminates.')
    else:
        if SAVE_MODEL_AND_DATA: save_model(current_date_model, encoders, model)
        switch_traffic = True    # default value for `switch_traffic` in case there are downstream issues with comparing the newly trained model with the currently deployed model
        try:
            mae_df_list = [current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df, business_date_before_test_data_date_data_previous_business_date_model_result_df]
            description_list = [f'The below table shows the accuracy of the newly trained {model} model for the trades that occurred on {test_data_date}.', 
                                f'The below table shows the accuracy of the {model} model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred on {test_data_date}. If there are three tables in this email, then this one evaluates on the same test dataset as the first table but with a different (previous business day) model. If the accuracy on this table is better than the first table, this may imply that the older model is more accurate. Note, however, that the model has not been (and, cannot be) evaluated yet on the trades that will occur today.', 
                                f'The below table shows the accuracy of the {model} model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred on {business_date_before_test_data_date}. If there are three tables in this email, then this one evaluates the same model as the second table but on a different (previous business day) test dataset. If the accuracy on this table is better than the second table, this may mean that the trades in the test set used for the first two tables are more challenging (harder to predict) than the trades from the test set used for this table.']
            mae_df_list, description_list = list(zip(*[(mae_df, description) for (mae_df, description) in zip(mae_df_list, description_list) if mae_df is not None]))    # only keep the (`mae_df`, `description`) pair if the `mae_df` is not None, and then put them into separate lists
            
            newly_trained_model_mae = current_date_data_current_date_model_result_df.loc[ROW_NAME_DETERMINING_MODEL_SWITCH, 'Mean Absolute Error']
            currently_deployed_model_mae = current_date_data_previous_business_date_model_result_df.loc[ROW_NAME_DETERMINING_MODEL_SWITCH, 'Mean Absolute Error']
            switch_traffic = newly_trained_model_mae <= currently_deployed_model_mae
            if switch_traffic:
                email_intro_text_addendum = f'{ROW_NAME_DETERMINING_MODEL_SWITCH} MAE of newly trained model ({newly_trained_model_mae}) is less than or equal to that of the currently deployed model ({currently_deployed_model_mae}) and so model traffic <b>has been switched</b> to the newly trained model.'
            else:
                email_intro_text_addendum = f'{ROW_NAME_DETERMINING_MODEL_SWITCH} MAE of newly trained model ({newly_trained_model_mae}) is greater than that of the currently deployed model ({currently_deployed_model_mae}) and so model traffic <b>has NOT been switched</b> to the newly trained model. All traffic remains on the currently deployed model.'
            training_logs_location = f'The training logs can be found in the Google Cloud Storage bucket: {BUCKET_NAME}, inside the directory: {TRAINING_LOGS_DIRECTORY}/.'
            send_results_email_multiple_tables(mae_df_list, description_list, current_date, EMAIL_RECIPIENTS, model, email_intro_text_addendum + '<hr>' + training_logs_location + '<hr>' + email_intro_text)    # use '<hr>' for the horizontal rule since this will create a horizontal line in the HTML body between the addendum and the other intro text
        except Exception as e:
            print(f'Switching traffic to the newly trained model since there may have been an issue with comparing the accuracy of the newly trained model with the currently deployed model. There may not be a currently deployed model within the last {MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK} days')
            print(f'{type(e)}:', e)
        return switch_traffic


def apply_exclusions(data: pd.DataFrame, dataset_name: str = None):
    print(f'Applying the exclusions function defined in `apply_exclusions(...)` now ...')
    from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
    data_before_exclusions = data[:]
    
    previous_size = len(data)
    data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_call <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_refund <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_maturity <= 400')
    
    previous_size = current_size
    data = data[data.days_to_maturity < np.log10(30000)]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having days_to_maturity >= 30000')
    
    ## null last_calc_date exclusion was removed on 2024-02-19
    # previous_size = current_size
    # data = data[~data.last_calc_date.isna()]
    # current_size = len(data)
    # if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having a null value for last_calc_date')

    return data, data_before_exclusions


def send_email(sender_email: str, message: str, recipients: list) -> None:
    '''Send email with `message` to `recipients` from `sender_email`.'''
    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            sender_password = 'ztwbwrzdqsucetbg'
            server.login(sender_email, sender_password)
            for receiver in recipients:
                server.sendmail(sender_email, receiver, message.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


def _get_email_subject(model_train_date: str, model: str) -> str:
    testing_addendum = '' if not TESTING else 'TESTING: '
    return f'(v2) {testing_addendum}MAE for {model} model trained on {model_train_date}'


def send_results_email(mae, model_train_date: str, recipients: list, model: str) -> None:
    check_that_model_is_supported(model)
    print(f'Sending email to {recipients}')
    
    msg = MIMEMultipart()
    msg['Subject'] = _get_email_subject(model_train_date, model)
    msg['From'] = SENDER_EMAIL

    message = MIMEText(f'MAE for {model} model on trades that occurred on {model_train_date} is {np.round(mae, 3)}.', 'plain')
    msg.attach(message)
    send_email(SENDER_EMAIL, msg, recipients)


def send_results_email_table(result_df, model_train_date: str, recipients: list, model: str) -> str:
    check_that_model_is_supported(model)
    print(f'Sending email to {recipients}')
    msg = MIMEMultipart()
    msg['Subject'] = _get_email_subject(model_train_date, model)
    msg['From'] = SENDER_EMAIL

    html_table = result_df.to_html(index=True)
    body = MIMEText(html_table, 'html')
    msg.attach(body)
    send_email(SENDER_EMAIL, msg, recipients)
    return html_table


def send_results_email_multiple_tables(df_list: list, text_list: list, model_train_date: str, recipients: list, model: str, intro_text: str = '') -> str:
    check_that_model_is_supported(model)
    print(f'Sending email to {recipients}')
    msg = MIMEMultipart()
    msg['Subject'] = _get_email_subject(model_train_date, model)
    msg['From'] = SENDER_EMAIL

    def html_text_for_single_df(text, df):
        return f'''{text}<br>{df.to_html(index=True)}'''

    if intro_text != '': intro_text = intro_text + '<hr>'
    html_text = f'''
    <html>
    <body>
    {intro_text}
    {'<hr>'.join([html_text_for_single_df(text, df) for text, df in zip(text_list, df_list)])}
    </body>
    </html>
    '''
    body = MIMEText(html_text, 'html')
    msg.attach(body)
    send_email(SENDER_EMAIL, msg, recipients)
    return html_text


def send_no_new_model_email(last_trade_date: str, recipients: list, model: str) -> None:
    check_that_model_is_supported(model)
    print(f'Sending email to {recipients}')
    msg = MIMEMultipart()
    next_week_day = increment_week_days(last_trade_date, 1)
    next_week_day_is_a_holiday = is_a_holiday(next_week_day)
    tag = f'{next_week_day} is a holiday so we do not expect new trades.' if next_week_day_is_a_holiday else f'ERROR: {next_week_day} is NOT a holiday so we expect new trades.'
    subject_prefix = f'{tag} Not enough new data was found on {next_week_day}'
    subject_suffix = f', so no new {model} model was trained'
    msg['Subject'] = f'{subject_prefix}{subject_suffix}'
    msg['From'] = SENDER_EMAIL
    html_text = f'''
    <html>
    <body>
    {subject_prefix} (the business day after {last_trade_date}){subject_suffix}; need at least {MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY} new trades to train a new model
    <hr>
    If the error is unexpected, perform the following procedure:
    <br>
    1. Check `{PROJECT_ID}.{AUXILIARY_VIEWS_DATASET_NAME}.trade_history_same_issue_5_yr_mat_bucket_1_materialized` to see if there are any trades from the most recent business date. If there are no trades, then a likely cause is that the S&P index data did not load correctly from the `update_sp_all_indices_and_maturities` cloud function.
    <br>
    2. Debug the `update_sp_all_indices_and_maturities` cloud function by inspecting the logs.
    <br>
    3. Follow the order of the following cloud functions below, and force run them to recover from the lost data. When force running the `compute_shape_parameter` cloud function, first update the `CURRENT_DATETIME` to be the previous business day if fixing it the next day.
    <br>
    4. Go to GCP scheduled queries for the `{PROJECT_ID}.{AUXILIARY_VIEWS_DATASET_NAME}.trade_history_same_issue_5_yr_mat_bucket_1_materialized` that is used for training. Click edit the scheduled query. This will open the query in a new window and you need simply click “Run” and let the query run for ~20 mins and the table will be ready. This will not actually edit or change the scheduled query.
    <br>
    5. Train the models by going into the VM, update your user using these instructions: https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8?pvs=4#463a8cb282e2454db42584317a31a42b. Then, run the corresponding command from https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8?pvs=4#122eb87466c28077b8b9d87f9f9490ec.
    <hr>
    See [this Notion page](https://www.notion.so/Yield-Curve-0e9d3fb1a49a4789826083361257a962?pvs=4#189eb87466c280d9ad01dc717ba0c6ae) for more details on related cloud functions and procedures. The training logs can be found in the Google Cloud Storage bucket: {BUCKET_NAME}, inside the directory: {TRAINING_LOGS_DIRECTORY}/.
    </body>
    </html>
    '''
    body = MIMEText(html_text, 'html')
    msg.attach(body)
    send_email(SENDER_EMAIL, msg, recipients)
