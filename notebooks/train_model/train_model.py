'''
Breakdown:
1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
2. python -u train_model.py: This part is executing your Python script in unbuffered mode, forcing Python to write output immediately.
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
import os
import sys
import pickle

import numpy as np
import pandas as pd


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'automated_training'))    # get the directory containing the 'ficc_python/automated_training' directory
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from auxiliary_variables import MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, BUCKET_NAME

import auxiliary_functions
auxiliary_functions.SAVE_MODEL_AND_DATA = False

from auxiliary_functions import train_model, setup_gpus, get_optional_arguments_for_process_data, get_data_and_last_trade_datetime    #, apply_exclusions
from clean_training_log import remove_lines_with_tensorflow_progress_bar


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ficc'))    # get the directory containing the 'ficc_python/ficc' directory
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from utils.auxiliary_functions import function_timer, get_ys_trade_history_features, get_dp_trade_history_features


MODEL = 'yield_spread_with_similar_trades'
NUM_DAYS = 5

TESTING = False
if TESTING:
    auxiliary_functions.NUM_EPOCHS = 4
    NUM_DAYS = 1


def restrict_trades_by_trade_datetime(df: pd.DataFrame, 
                                      start_trade_datetime: str = None, 
                                      end_trade_datetime: str = None) -> pd.DataFrame:
    '''`start_trade_datetime` and `end_trade_datetime` can be string objects representing a date, e.g., '2025-01-17' 
    because numpy automatically converts the string into a datetime object before making the comparison.'''
    if start_trade_datetime is not None: df = df[df['trade_datetime'] >= start_trade_datetime]
    if end_trade_datetime is not None: df = df[df['trade_datetime'] <= end_trade_datetime]
    return df


@function_timer
def get_processed_data_pickle_file(model: str = MODEL) -> pd.DataFrame:
    file_name = MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[model]
    if os.path.isfile(file_name):
        print(f'Loading data from {file_name} which was found locally...')
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        most_recent_trade_datetime = data.trade_datetime.max()
    else:
        print(f'Did not find {file_name} locally so downloading it from Google Cloud Storage...')
        data, most_recent_trade_datetime, _ = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        file_name = f'gs://{BUCKET_NAME}/processed_data/{file_name}'    # used for print
    print(f'Loaded data from {file_name}. Most recent trade datetime: {most_recent_trade_datetime}')
    return data


def get_num_features_for_each_trade_in_history(model: str = MODEL) -> int:
    optional_arguments = get_optional_arguments_for_process_data(model)
    use_treasury_spread = optional_arguments.get('use_treasury_spread', False)    # from `auxiliary_functions.py::update_data(...)`
    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if 'yield_spread' in model else get_dp_trade_history_features()    # from `automated_training/auxiliary_functions.py::get_new_data(...)`
    return len(trade_history_features)    # from `auxiliary_functions.py::get_new_data(...)`


def train_model_from_data_file(data: pd.DataFrame, num_days: int, output_file_path: str = None, exclusions_function: callable = None):
    most_recent_dates = np.sort(data['trade_date'].unique())[::-1]    # sort the unique `trade_date`s in descending order (the descending order comes from the slice)
    most_recent_dates = most_recent_dates[:num_days + 1]    # restrict to `num_days` most recent dates
    for day_idx in range(num_days):
        date_for_test_set, most_recent_date_for_training_set = most_recent_dates[day_idx], most_recent_dates[day_idx + 1]
        data = data[data['trade_date'] <= date_for_test_set]    # iteratively remove the last date from `data`
        trained_model, _, _, _, _, mae, (mae_df, _), _ = train_model(data, most_recent_date_for_training_set, MODEL, get_num_features_for_each_trade_in_history(), exclusions_function=exclusions_function)
        if output_file_path is not None: remove_lines_with_tensorflow_progress_bar(output_file_path)
        trained_model.save(f'{MODEL}_{date_for_test_set}')
        

if __name__ == '__main__':
    setup_gpus(False)
    data = get_processed_data_pickle_file(MODEL)
    output_file_name = 'output.txt'
    train_model_from_data_file(data, NUM_DAYS, output_file_name)    # , exclusions_function=apply_exclusions)