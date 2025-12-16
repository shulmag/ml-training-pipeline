'''
Description: This file contains function to help the functions to process training data
'''
import os
import time
from functools import wraps
from datetime import datetime, timedelta
import warnings
import pickle

import urllib3
import requests

import pandas as pd

from google.api_core.exceptions import BadRequest

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR, YS_BASE_TRADE_HISTORY_FEATURES, DP_BASE_TRADE_HISTORY_FEATURES
from ficc.utils.diff_in_days import diff_in_days_two_dates


def remove_fractional_seconds_beyond_3_digits(original_timedelta):
    return str(original_timedelta)[:-3]    # total of 6 digits after the decimal, so we keep everything but the last 3


def function_timer(function_to_time):
    '''This function is to be used as a decorator. It will print out the execution time of `function_to_time`.'''
    @wraps(function_to_time)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
    def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
        print(f'BEGIN {function_to_time.__name__}')
        start_time = time.time()
        result = function_to_time(*args, **kwargs)
        end_time = time.time()
        time_elapsed = timedelta(seconds=end_time - start_time)
        print(f'END {function_to_time.__name__}. Execution time: {remove_fractional_seconds_beyond_3_digits(time_elapsed)}')    # remove microseconds beyond 3 decimal places since this level of precision is unnecessary and just adds noise
        return result
    return wrapper


def run_multiple_times_before_failing(error_types: tuple, max_runs: int = 10, exponential_backoff: bool = False):
    '''This function returns a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a maximum of `max_runs` times. If `exponential_backoff` is set to `True`, then 
    the wait time is increased exponentially, otherwise it is a constant value.
    NOTE: max_runs = 1 is the same functionality as not having this decorator.'''
    def decorator(function):
        @wraps(function)    # used to ensure that the function name is still the same after applying the decorator when running tests: https://stackoverflow.com/questions/6312167/python-unittest-cant-call-decorated-test
        def wrapper(*args, **kwargs):    # using the same formatting from https://docs.python.org/3/library/functools.html
            runs_so_far = 0
            while runs_so_far < max_runs:    # NOTE: max_runs = 1 is the same functionality as not having this decorator
                try:
                    return function(*args, **kwargs)
                except error_types as e:
                    runs_so_far += 1
                    if runs_so_far >= max_runs:
                        warnings.warn(f'Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')    # python_logging.warning(f'Already caught {type(e)}: {e}, {max_runs} times in {function.__name__}, so will now raise the error')
                        raise e
                    warnings.warn(f'WARNING: Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')    # python_logging.warning(f'Caught {type(e)}: {e}, and will retry {function.__name__} {max_runs - runs_so_far} more times')
                    delay = min(2 ** (runs_so_far - 1), 10) if exponential_backoff else 1
                    time.sleep(delay)    # have a delay to prevent overloading the server
        return wrapper
    return decorator


def run_ten_times_before_raising_gcp_bucket_access_error(function: callable) -> callable:
    '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a specified number of times. It solves the following problem: GCP limits how quickly 
    files can be downloaded from buckets and raises an `SSLError` or a `KeyError` when the buckets are accessed 
    too quickly in succession.'''
    return run_multiple_times_before_failing((KeyError, urllib3.exceptions.SSLError, requests.exceptions.SSLError), 10)(function)    # catches KeyError: 'email', KeyError: 'expires_in', urllib3.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad, requests.exceptions.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac 


def run_ten_times_before_raising_bigquery_error(function: callable) -> callable:
    '''This function is to be used as a decorator. It will run `function` over and over again until it does not 
    raise an Exception for a specified number of times. It solves the following problem: BigQuey has transient 
    issues when querying a table.'''
    return run_multiple_times_before_failing((BadRequest,), 10)(function)


def double_quote_a_string(potential_string):
    '''Quote a string twice: e.g., double_quote_a_string('hello') -> "'hello'". This 
    function is used to put string arguments into formatted string expressions and 
    maintain the quotation.'''
    return f'"{str(potential_string)}"' if type(potential_string) == str else potential_string


@run_ten_times_before_raising_bigquery_error
def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            print('************ ERROR ************')
            print(f'Failed to convert date for {col}')
            print('*******************************')
            print('Error:', e)
            continue
    
    return df


def process_ratings(df):
    original_rating_column_name = 'sp_long'
    null_rating_column = df[original_rating_column_name].isnull()
    assert not null_rating_column.all(), f'Ratings data (`{original_rating_column_name}`) is missing from the dataframe (all values are null)'
    df[original_rating_column_name].fillna('MR', inplace=True)    # MR is for missing ratings
    print(f'Filled {original_rating_column_name} with "MR" for {null_rating_column.sum()} null rows')
    df['rating'] = df[original_rating_column_name]
    return df


def convert_to_date(date):
    '''Converts an object, either of type pd.Timestamp or datetime.datetime to a 
    datetime.date object.'''
    if isinstance(date, pd.Timestamp): date = date.to_pydatetime()
    if isinstance(date, datetime): date = date.date()
    return date    # assumes the type is datetime.date


def compare_dates(date1, date2):
    '''Compares two date objects whether they are in Timestamp or datetime.date. 
    The different types are causing a future warning. If date1 occurs after date2, return 1. 
    If date1 equals date2, return 0. Otherwise, return -1.'''
    return (convert_to_date(date1) - convert_to_date(date2)).total_seconds()


def dates_are_equal(date1, date2):
    '''Directly calls `compare_dates` to check if two dates are equal.'''
    return compare_dates(date1, date2) == 0

def convert_object_to_category(df):
    '''Converts the columns with object datatypes to category data types'''
    print('Converting object data type to categorical data type')
    for col_name in df.columns:
        if col_name.endswith('event') or col_name.endswith('redemption') or col_name.endswith('history') or col_name.endswith('date') or col_name.endswith('issue'):
            continue

        if df[col_name].dtype == 'object' and col_name not in ['organization_primary_name','security_description','recent','issue_text','series_name','recent_trades_by_series','recent_trades_same_calc_day']:
            df[col_name] = df[col_name].astype('category')
    return df


def calculate_a_over_e(df):
    if not pd.isnull(df.previous_coupon_payment_date):
        A = (df.settlement_date - df.previous_coupon_payment_date).days
        return A / df.days_in_interest_payment
    else:
        return df['accrued_days'] / NUM_OF_DAYS_IN_YEAR


def calculate_dollar_error(df, predicted_ys):
    '''Computes the dollar error from the predicted yield spreads and MSRB data. 
    Assumes that the predicted yield spreads are in basis points.'''
    assert len(predicted_ys) == len(df), 'There must be a predicted yield spread for each of the trades in the passed in dataframe'
    columns_set = set(df.columns)
    assert 'quantity' in columns_set    # assumes that the quantity is log10 transformed
    assert 'new_ficc_ycl' in columns_set    # represents the ficc yield curve level
    assert 'yield' in columns_set    # represents yield to worst from the MSRB data
    assert 'calc_date' in columns_set and 'settlement_date' in columns_set    # need these two features to compute the number of years from the settlement date to the calc date
    years_to_calc_date = diff_in_days_two_dates(df.calc_date, df.settlement_date) / NUM_OF_DAYS_IN_YEAR    # the division by `np.timedelta64(NUM_OF_DAYS_IN_YEAR, 'D')` converts the quantity to years according to the MSRB convention of NUM_OF_DAYS_IN_YEAR in a year
    ytw_error = ((predicted_ys + df['new_ficc_ycl']) / 100 - df['yield']) / 100    # the second divide by 100 is because the unit of the dividend is in percent
    return ytw_error * (10 ** df['quantity']) * years_to_calc_date    # dollar error = duration * quantity * ytw error; duration = calc_date - settlement_date [in years]


def get_ys_trade_history_features(treasury_spread=False):
    if treasury_spread:
        return YS_BASE_TRADE_HISTORY_FEATURES[:1] + ['treasury_spread'] + YS_BASE_TRADE_HISTORY_FEATURES[1:]
    return YS_BASE_TRADE_HISTORY_FEATURES


def get_dp_trade_history_features():
    return DP_BASE_TRADE_HISTORY_FEATURES


def check_if_pickle_file_exists_and_matches_query(query: str, path: str) -> pd.DataFrame | None:
    '''Checks if the pickle file exists and if it matches the query. If it does, it loads the data from the pickle file. 
    Otherwise, it returns None.'''
    if os.path.isfile(path):
        print(f'Data file {path} found, reading data from it')
        with open(path, 'rb') as f: 
            (query_in_file, df) = pickle.load(f)
        if query_in_file == query:
            return df
        else:
            print('Query does not match. Returning None')
            return None    # raise Exception(f'Saved query is incorrect:\n{q}')
    else:
        print(f'Data file {path} not found, returning None')
        return None
