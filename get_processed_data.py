**NOTE**: Set the `TESTING` flag to `True` if just testing the data generation procedure.
**NOTE**: Set credentials appropriately in `automated_training/auxiliary_functions.py::get_creds()`.
**NOTE**: To get an entire month of trades using 32 CPUs, set the memory on the VM to 250 GB.
**NOTE**: To see the output of this script in an `output.txt` file use the command: $ python -u get_processed_data.py >> output.txt. `-u` ensures that the text is immediately written to the output file instead of waiting for the entire procedure to complete.
**NOTE**: To run the procedure in the background, use the command: $ nohup python -u get_processed_data.py >> output.txt 2>&1 &. This will return a process number such as [1] 66581, which can be used to kill the process.
Breakdown:
1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
2. `python -u get_processed_data.py`: This part is executing your Python script in unbuffered mode, forcing Python to write output immediately.
3. >> output.txt 2>&1:
    * >> output.txt appends the standard output (stdout) of the script to output.txt instead of overwriting it.
    * 2>&1 redirects standard error (stderr) to the same file as standard output, so both stdout and stderr go into output.txt.
4. &: This runs the command in the background.

To redirect the error to a different file, you can use 2> error.txt. Note that just ignoring it (not including 2>...) will just output to std out in this case.

To kill the command, run
$ kill 66581
or
$ kill -9 66581
The -9 forces the operation.
'''
import os    # used for `os.cpu_count()` when setting the number of workers in `mp.Pool()`
import sys
import pickle
from datetime import datetime

from tqdm import tqdm
import multiprocess as mp
import pandas as pd

from ficc.utils.auxiliary_functions import sqltodf, function_timer, check_if_pickle_file_exists_and_matches_query

from automated_training.auxiliary_variables import EASTERN, BUSINESS_DAY, YEAR_MONTH_DAY, EARLIEST_TRADE_DATETIME, MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, YEAR_MONTH_DAY
from automated_training.auxiliary_functions import BQ_CLIENT, get_data_query, check_that_model_is_supported, get_new_data, get_optional_arguments_for_process_data, combine_new_data_with_old_data, add_trade_history_derived_features, drop_features_with_null_value, save_data


MODEL = 'yield_spread_with_similar_trades'
check_that_model_is_supported(MODEL)

MULTIPROCESSING = True
SAVE_DATA = True

TESTING = False
if TESTING:
    SAVE_DATA = False
    EARLIEST_TRADE_DATETIME = (datetime.now(EASTERN) - (BUSINESS_DAY * 2)).strftime(YEAR_MONTH_DAY) + 'T00:00:00'    # 2 business days before the current datetime (start of the day) to have enough days for training and testing; same logic as `auxiliary_functions::decrement_business_days(...)` but cannot import from there due to circular import issue


@function_timer
def check_that_df_is_sorted_by_column(df, column_name, desc: bool = False):
    '''`desc` is a boolean flag that determines whether we are checking for ascending order 
    (`desc` is `False`) or descending order (`desc` is `True`).'''
    column_values = df[column_name].to_numpy()
    larger = column_values[:-1] if desc else column_values[1:]
    smaller = column_values[1:] if desc else column_values[:-1]
    ascending_or_descending = 'descending' if desc else 'ascending'
    assert all(larger >= smaller), f'Dataframe should be sorted in {ascending_or_descending} order by column name: {column_name}, but is not'


@function_timer
def get_processed_trades_for_particular_date(start_date_as_string: str, end_date_as_string: str = None, use_multiprocessing: bool = True) -> pd.DataFrame:
    '''If `end_date_as_string` is `None`, then we get trades only for `start_date_as_string`.'''
    if end_date_as_string is None: end_date_as_string = start_date_as_string
    print(f'Start date: {start_date_as_string}\t\tEnd date: {end_date_as_string}')
    data_query = get_data_query(start_date_as_string + 'T00:00:00', MODEL)
    order_by_position = data_query.find('ORDER BY')
    data_query_date = data_query[:order_by_position] + f'AND trade_datetime <= "{end_date_as_string}T23:59:59" ' + data_query[order_by_position:]    # add condition of restricting all trades to the specified `date_as_string`
    # print(data_query_date)    # this gets printed inside `fetch_trade_data(...)`

    file_name = f'processed_data_for_date_{start_date_as_string}.pkl'
    if TESTING:
        processed_data = check_if_pickle_file_exists_and_matches_query(data_query_date, file_name)
        if processed_data is not None:
            return processed_data

    optional_arguments_for_process_data = get_optional_arguments_for_process_data(MODEL)
    use_treasury_spread = optional_arguments_for_process_data.get('use_treasury_spread', False)
    _, processed_data, _, _, _ = get_new_data(None, MODEL, use_treasury_spread, optional_arguments_for_process_data, data_query_date, SAVE_DATA, use_multiprocessing, f'{os.path.dirname(__file__)}/files/raw_data_{start_date_as_string}.pkl')
    
    if TESTING:
        # save `processed_data` to a pickle file for re-using later
        if os.path.isfile(file_name):
            print(f'File {file_name} already exists. Deleting it.')
            os.remove(file_name)
        with open(file_name, 'wb') as file:
            pickle.dump((data_query_date, processed_data), file)
    
    return processed_data    # get just the raw data with `sqltodf(data_query_date, BQ_CLIENT)`


@function_timer
def get_trades_for_all_dates(dates: list, 
                             file_name: str = MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[MODEL], 
                             all_at_once: bool = False) -> pd.DataFrame:
    '''`dates` is a list of strings.'''
    if all_at_once is True:
        trades_for_all_dates = get_processed_trades_for_particular_date(min(dates), max(dates))
    else:
        if len(dates) > 1 and MULTIPROCESSING:
            num_workers = os.cpu_count()    # consider trying with lesser processes if running out of RAM (e.g., set `num_workers` to `os.cpu_count() // 4`)
            print(f'Using multiprocessing with {num_workers} workers for calling `get_trades_for_particular_date(...) on each of the {len(dates)} items in {dates}')
            if sys.platform == "darwin":    # macOS; sys.platform returns "darwin" for macOS, "linux" for Linux, "win32" for Windows
                mp.set_start_method("spawn", force=True)    # when on macOS safer, especially when GUI/network/system calls are involved, to use `spawn` instead of `fork` to avoid issues with pickling the function and its arguments; see https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods; spawn starts a fresh Python interpreter process instead of duplicating the parent process memory via fork; `force=True` forces the start method to be changed even if it was already set before
            with mp.Pool(num_workers) as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                trades_for_all_dates = pool_object.map(lambda start_date_as_string: get_processed_trades_for_particular_date(start_date_as_string, use_multiprocessing=False), dates)
        else:
            trades_for_all_dates = [get_processed_trades_for_particular_date(date) for date in tqdm(dates, disable=len(dates) == 1)]
        trades_for_all_dates = pd.concat(trades_for_all_dates).reset_index(drop=True)    # `pd.concat(...)` is a bottleneck (even though the calls to `get_trades_for_particular_date(...)` take less than a minute, the `pd.concat(...)` takes >10 mins
    check_that_df_is_sorted_by_column(trades_for_all_dates, 'trade_datetime', desc=True)

    optional_arguments_for_process_data = get_optional_arguments_for_process_data(MODEL)
    # next few lines are very similar to `ficc_python/auxiliary_functions.py::update_data(...)`
    use_treasury_spread = optional_arguments_for_process_data.get('use_treasury_spread', False)
    trades_for_all_dates = combine_new_data_with_old_data(None, trades_for_all_dates, MODEL, use_treasury_spread)
    trades_for_all_dates = add_trade_history_derived_features(trades_for_all_dates, MODEL, use_treasury_spread)
    trades_for_all_dates = drop_features_with_null_value(trades_for_all_dates, MODEL)
    if SAVE_DATA: save_data(trades_for_all_dates, file_name, upload_to_google_cloud_bucket=False)
    return trades_for_all_dates


def check_date_in_correct_format(date_as_string):
    '''Try to convert the date string to a datetime object. If the conversion fails, the date string is invalid.
    Generated by Claude with prompt: "check if date is in the right format python".'''
    try:
        datetime.strptime(date_as_string, YEAR_MONTH_DAY)
        return True
    except ValueError:
        return False


@function_timer
def create_data_for_start_end_date_pair(start_datetime: str,    # may be a string representation of a date instead of a datetime
                                        end_datetime: str,    # may be a string representation of a date instead of a datetime
                                        file_name: str = 'trades_for_all_dates_from_get_processed_data.pkl') -> pd.DataFrame:
    '''Create a file that contains the trades between `start_datetime` and `end_datetime`. Save the file 
    in a file with name: `file_name`.'''
    data_query = get_data_query(start_datetime, MODEL, end_datetime)
    distinct_dates_query = 'SELECT DISTINCT trade_date ' + data_query[data_query.find('FROM') : data_query.find('ORDER BY')]    # remove all the original selected features and just get each unique `trade_date`; need to remove the `ORDER BY` clause since the `trade_datetime` feature is not selected in this query
    print('Getting distinct dates from the following query:\n', distinct_dates_query)
    distinct_dates = sqltodf(distinct_dates_query, BQ_CLIENT)
    distinct_dates = sorted(distinct_dates['trade_date'].astype(str).values, reverse=True)    # convert the one column dataframe with column name `trade_date` from `sqltodf(...)` into a numpy array sorted by `trade_date`; going in descending order since the the query gets the trades in descending order of `trade_datetime` and so concatenating all the trades from each of the days will be in descending order of `trade_datetime` if the trade dates are in descending order
    print('Distinct dates:', distinct_dates)

    trades_for_all_dates = get_trades_for_all_dates(distinct_dates, file_name, False)
    # print(trades_for_all_dates)
    return trades_for_all_dates


def create_data_for_start_end_date_pairs(date_pairs: list) -> pd.DataFrame:
    '''Create a set of files where each file contains the trades for each date pair in `date_pairs`, 
    which is a list of pairs (tuples) where each pair contains the start date and end date.'''
    return pd.concat([create_data_for_start_end_date_pair(start_date, end_date, f'trades_{start_date}_to_{end_date}.pkl') for start_date, end_date in date_pairs])


@function_timer
def main():
    '''Get the trades for all dates in the date range. The date range is specified by the command line arguments.
    The first argument is the latest trade date. The second argument is the earliest trade date. If no arguments are provided,
    the default values are used. The default values are:
    1. `latest_trade_date`: `None`
    2. `earliest_trade_datetime`: `EARLIEST_TRADE_DATETIME`
    The `earliest_trade_datetime` is the earliest trade date that is available in the database. The `latest_trade_date` is the latest trade date that is available in the database.
    
    Example usage:
    $ python get_processed_data.py 2025-01-17 2025-01-15
    $ python get_processed_data.py 2025-01-17
    $ python get_processed_data.py'''
    latest_trade_date = sys.argv[1] if len(sys.argv) >= 2 else None
    earliest_trade_datetime = sys.argv[2] if len(sys.argv) >= 3 else EARLIEST_TRADE_DATETIME    # create this variable to easily modify the value instead of trying to modify `EARLIEST_TRADE_DATETIME` which gives `UnboundLocalError: local variable 'EARLIEST_TRADE_DATETIME' referenced before assignment`
    if latest_trade_date is not None:
        assert check_date_in_correct_format(latest_trade_date)
        if TESTING and earliest_trade_datetime == EARLIEST_TRADE_DATETIME:
            earliest_trade_datetime = (datetime.strptime(latest_trade_date, YEAR_MONTH_DAY) - (BUSINESS_DAY * 2)).strftime(YEAR_MONTH_DAY) + 'T00:00:00'    # 2 business days before the current datetime (start of the day) to have enough days for training and testing; same logic as `auxiliary_functions::decrement_business_days(...)` but cannot import from there due to circular import issue
    return create_data_for_start_end_date_pair(earliest_trade_datetime, latest_trade_date)


if __name__ == '__main__':
    trades_for_all_dates = main()
