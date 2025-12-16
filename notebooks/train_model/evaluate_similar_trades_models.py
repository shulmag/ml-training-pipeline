'''
MAJOR UPDATE (July 2025): Extended from a training-only script to include:
- Historical model evaluation capabilities
- Production-style backtesting across all models
- RMSE metrics alongside MAE
- Enhanced BigQuery integration for prediction storage
- Comprehensive performance reporting

USAGE MODES:
1. train (default): Train a new model on recent data
    To train multiple models, modify the NUM_DAYS variable. This trains NUM_DAYS number of models over a sliding window
    over the NUM_DAYS most recent days in the data pickle file. By default, this outputs a CSV with MAE for the full set of trades
    on each test date, as well as the exclusions_applied test set. Easily add your own test set of data in the train_model_from_data_file function. 
   $ python train_model.py

2. list: Show all available models in GCS
   $ python train_model.py --mode list

3. evaluate: Test specific model(s) on specific date(s)
   $ python train_model.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04

4. production-eval: Comprehensive backtesting - each model tested on all its production days
   $ python train_model.py --mode production-eval
   $ python train_model.py --mode production-eval --start-date 2025-03-01 --end-date 2025-03-31

**NOTE**: BACKGROUND EXECUTION:
$ nohup python -u train_model.py --mode production-eval >> output.txt 2>&1 &
This returns a process number (e.g., [1] 66581) that can be used to monitor or kill the process.
To run the procedure in the background, use the command: $ nohup python -u train_model.py >> output.txt 2>&1 &. This will return a process number such as [1] 66581, which can be used to kill the process.
Breakdown:
1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
2. python -u train_model.py: This part is executing your Python script in unbuffered mode, forcing Python to write output immediately.
3. >> output.txt 2>&1:
    * >> output.txt appends the standard output (stdout) of the script to output.txt instead of overwriting it.
    * 2>&1 redirects standard error (stderr) to the same file as standard output, so both stdout and stderr go into output.txt.
4. &: This runs the command in the background.

To monitor: $ tail -f output.txt
To kill: $ kill 66581 (or kill -9 66581 to force)

OUTPUTS:
- Trained models: Saved locally and to GCS
- Predictions: Uploaded to BigQuery sandbox table
- Summary CSVs: Performance metrics by model and days since training
- Console output: Detailed evaluation results

See README.md for more detailed documentation.

To train a model with a processed data file: Heavily uses code from `automated_training/`. Note: update `auxiliary_functions.py::get_creds(...)` with the correct file path.

To redirect the error to a different file, you can use 2> error.txt. Note that just ignoring it (not including 2>...) will just output to std out in this case.

'''

import os
import sys
import pickle

import numpy as np
import pandas as pd
from pathlib import Path

from google.cloud import storage
import re
from datetime import datetime

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/gil/git/ficc/creds.json"

# 1) Put the repo root on sys.path ONCE
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

# 2) Use package-qualified imports (no bare names)
from automated_training.auxiliary_variables import MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, BUCKET_NAME
import automated_training.auxiliary_functions as aux
from automated_training.clean_training_log import remove_lines_with_tensorflow_progress_bar

# 3) All calls go through the aux namespace (single source of truth)
aux.SAVE_MODEL_AND_DATA = True
# e.g., model, *_ = aux.train_model(...)

# 4) utils under the ficc package (qualified)
from ficc.utils.auxiliary_functions import (
    function_timer, get_ys_trade_history_features, get_dp_trade_history_features
)



MODEL = 'yield_spread_with_similar_trades'
NUM_DAYS = 1

TESTING = False
if TESTING:
    aux.auxiliary_functions.NUM_EPOCHS = 4
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
def get_processed_data_pickle_file(date: str = None, model: str = MODEL) -> pd.DataFrame:
    file_name = "/Users/gil/git/ficc/notebooks/gil_modeling/embeddings/processed_data_yield_spread_with_similar_trades_v2.pkl"  # MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[model]
    if os.path.isfile(file_name):
        print(f'Loading data from {file_name} which was found locally...')
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        if date is not None:
            data = data[data['trade_date'] <= date]
        most_recent_trade_datetime = data.trade_datetime.max()

    else:
        raise NotImplementedError
        print(f'Did not find {file_name} locally so downloading it from Google Cloud Storage...')
        data, most_recent_trade_datetime, _ = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        file_name = f'gs://{BUCKET_NAME}/processed_data/{file_name}'    # used for print
    print(f'Loaded data from {file_name}. Most recent trade datetime: {most_recent_trade_datetime}')
    return data


def get_num_features_for_each_trade_in_history(model: str = MODEL) -> int:
    optional_arguments = aux.get_optional_arguments_for_process_data(model)
    use_treasury_spread = optional_arguments.get('use_treasury_spread', False)    # from `auxiliary_functions.py::update_data(...)`
    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if 'yield_spread' in model else get_dp_trade_history_features()    # from `automated_training/auxiliary_functions.py::get_new_data(...)`
    return len(trade_history_features)    # from `auxiliary_functions.py::get_new_data(...)`


def evaluate_with_model(keras_model, test_df: pd.DataFrame, encoders) -> dict:
    """Run model on test_df, compute MAE/RMSE using your existing helpers."""
    if test_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "trade_count": 0}

    x_test, y_test = aux.create_input(test_df, encoders, MODEL)
    preds = keras_model.predict(x_test, batch_size=1000)
    delta = np.abs(preds.flatten() - y_test)
    res_df = aux.segment_results(test_df, delta)

    mae = float(res_df.loc["Entire set", "Mean Absolute Error"])
    rmse = float(res_df.loc["Entire set", "RMSE"]) if "RMSE" in res_df.columns else np.nan
    trade_count = int(res_df.loc["Entire set", "Trade Count"])

    return {"mae": mae, "rmse": rmse, "trade_count": trade_count}

def load_encoders(encoders_path: str):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    with fs.open(encoders_path, "rb") as f:
        return pickle.load(f)
    print(f"Loaded encoders from {encoders_path}")

def train_model_from_data_file(data: pd.DataFrame, num_days: int, output_file_path: str = None, exclusions_function: callable = None):
    import gc
    import tensorflow as tf
    #To test the model on specific test data, add your data's path here, and load the dataframe for your data once, outside the num_days loop
    #Ex:
    # MY_TEST_SET = "~/ficc_python/notebooks/test_data/test_rtrs_cntrl_nums.csv"
    # test_set = set(pd.read_csv(MY_TEST_SET))
    ENCODERS_GCS_PATH = "gs://automated_training/encoders_similar_trades.pkl"
    encoders   = load_encoders(ENCODERS_GCS_PATH)

    
    most_recent_dates = np.sort(data['trade_date'].unique())[::-1]    # sort the unique `trade_date`s in descending order (the descending order comes from the slice)
    most_recent_dates = most_recent_dates[:num_days + 1]    # restrict to `num_days` most recent dates
    
    # prepare outputs ONCE (outside the loop)
    out_csv = "model_results.csv"
    out_log = "model_log.log"
    with open(out_csv, "w") as f:
        f.write("model_train_date,test_date,test_set,mae,rmse,trade_count\n")
    Path(out_log).write_text("")

    for day_idx in range(num_days):

        test_date  = most_recent_dates[day_idx]
        train_date = most_recent_dates[day_idx + 1]
        window_df = data.loc[data['trade_date'] <= test_date]
        # train
        trained_model, _, _, _, _, mae, (mae_df, _), _ = aux.train_model(
            window_df,
            train_date,
            MODEL,
            get_num_features_for_each_trade_in_history(),
            exclusions_function=exclusions_function
        )
        
        if output_file_path is not None: 
            remove_lines_with_tensorflow_progress_bar(output_file_path)
        trained_model.save(f'{MODEL}_{test_date}')

        # build eval sets as lightweight views. Default evaluation sets are the full test day of trades,
        # and the test day filtered using the apply_exclusions function which filters long and short term maturities.
        # Add your custom test sets here if you wish to test the model on specific trades
        eval_sets = []
        day_df = data.loc[data["trade_date"] == test_date]
        eval_sets.append(("full test day", day_df))
        exclusions_set, _ = aux.apply_exclusions(day_df)
        eval_sets.append(("exclusions applied", exclusions_set))

        # stream results to disk
        with open(out_log, "a") as lf, open(out_csv, "a") as cf:
            for set_name, df in eval_sets:
                try:
                    metrics = evaluate_with_model(trained_model, df, encoders)
                    cf.write(f"{train_date},{test_date},{set_name},"
                             f"{metrics['mae']},{metrics['rmse']},{metrics['trade_count']}\n")
                    lf.write(f"  - {set_name:12s} | n={metrics['trade_count']:5d} | "
                             f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f}\n")
                except Exception as e:
                    lf.write(f"  - {set_name:12s} | EVAL ERROR: {e}\n")
        # ---- free memory for this iteration ----
        del day_df, exclusions_set, trained_model, mae_df, window_df
        tf.keras.backend.clear_session()
        gc.collect()


def list_similar_trades_models(bucket_name='automated_training', prefix='similar-trades-v2-model-'):
    """List all similar-trades-v2 models in GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    models = []
    seen_dates = set()  # To avoid duplicates
    
    for blob in bucket.list_blobs(prefix=prefix):
        # Extract date from model name (similar-trades-v2-model-YYYY-MM-DD/)
        match = re.search(r'similar-trades-v2-model-(\d{4}-\d{2}-\d{2})', blob.name)
        if match:
            date_str = match.group(1)
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                models.append({
                    'date': date_str,
                    'path': f'gs://{bucket_name}/similar-trades-v2-model-{date_str}/',
                    'name': f'similar-trades-v2-model-{date_str}'
                })
    
    # Sort by date
    models.sort(key=lambda x: x['date'], reverse=True)
    
    print(f"Found {len(models)} similar-trades-v2 models:")
    for model in models[:10]:  # Show first 10
        print(f"  {model['date']}: {model['path']}")
    
    return models

def evaluate_model_on_single_day(
    model_date: str,
    test_date: str,
    data: pd.DataFrame = None,
    data_pkl_path: str = None,
    upload_to_bq: bool = True,
    fixed_quantity: float = None,  # Add parameter for fixed quantity
    cycle_trade_types: bool = False  # Add flag to enable trade type cycling
):
    """Load a pre-trained model and evaluate it on a specific test date
    
    If cycle_trade_types is True, will create 3 predictions per trade (one for each trade type D, S, P)
    with fixed quantity specified by fixed_quantity parameter.
    """
    
    import tensorflow as tf
    from tensorflow import keras
    from datetime import datetime
    from pytz import timezone
    import gcsfs
    
    # 1. Load the data if not provided
    if data is None:
        if data_pkl_path is None:
            data_pkl_path = "/Users/gil/git/ficc/notebooks/gil_modeling/embeddings/processed_data_yield_spread_with_similar_trades_v2.pkl"
        print(f"Loading data from {data_pkl_path}")
        with open(data_pkl_path, 'rb') as f:
            data = pickle.load(f)
    
    # 2. Filter to test date
    test_data = data[data['trade_date'] == test_date]
    print(f"Test data shape for {test_date}: {test_data.shape}")
    
    if len(test_data) == 0:
        print(f"No data found for test date {test_date}")
        return None
    
    # MODIFICATION: Create expanded dataset with fixed quantity and cycling trade types
    if fixed_quantity is not None:
        if cycle_trade_types:
            # Create 3 copies of each trade, one for each trade type
            test_data_expanded = []
            trade_types = ['D', 'S', 'P']
            
            for _, row in test_data.iterrows():
                for trade_type in trade_types:
                    row_copy = row.copy()
                    row_copy['quantity'] = np.log10(fixed_quantity * 1000)  # Convert to log10 of par value
                    row_copy['par_traded'] = fixed_quantity * 1000  # Actual par value
                    row_copy['trade_type'] = trade_type
                    
                    # Update the target_attention_features to reflect the new quantity and trade type
                    # This is important because the model uses these features
                    # target_attention_features is created by target_trade_processing_for_attention
                    # which creates [quantity, trade_type_encoding]
                    trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
                    target_features = [row_copy['quantity']] + trade_mapping[trade_type]
                    row_copy['target_attention_features'] = np.tile(target_features, (1, 1))
                    
                    test_data_expanded.append(row_copy)
            
            test_data = pd.DataFrame(test_data_expanded)
            print(f"Expanded test data shape (3x for trade types): {test_data.shape}")
        else:
            # Just modify quantity for existing trades
            test_data = test_data.copy()
            test_data['quantity'] = np.log10(fixed_quantity * 1000)
            test_data['par_traded'] = fixed_quantity * 1000
            
            # Update target_attention_features for the new quantity
            for idx in test_data.index:
                trade_type = test_data.loc[idx, 'trade_type']
                trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
                target_features = [test_data.loc[idx, 'quantity']] + trade_mapping[trade_type]
                test_data.loc[idx, 'target_attention_features'] = np.tile(target_features, (1, 1))
    
    # 3. Load the model
    model_path = f'gs://automated_training/similar-trades-v2-model-{model_date}'
    # model_path = '~/ficc_python/notebooks/train_model/local_model_directory'
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # 4. Load the encoders from GCS
    fs = gcsfs.GCSFileSystem()
    encoders_path = 'gs://automated_training/encoders_similar_trades.pkl'
    with fs.open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    print(f"Loaded encoders from {encoders_path}")
    
    # 5. Create inputs
    # from auxiliary_functions import create_input
    x_test, y_test = aux.create_input(test_data, encoders, 'yield_spread_with_similar_trades')
    
    # 6. Generate predictions
    predictions = model.predict(x_test, batch_size=1000)
    
    # 7. Calculate metrics and segment results
    # from auxiliary_functions import segment_results
    delta = np.abs(predictions.flatten() - y_test)
    result_df = aux.segment_results(test_data, delta)
    
    print(f"\nResults for model {model_date} on test date {test_date}:")
    if fixed_quantity:
        if cycle_trade_types:
            print(f"(Using fixed quantity of {fixed_quantity}k and cycling through trade types D, S, P)")
        else:
            print(f"(Using fixed quantity of {fixed_quantity}k)")
    print(result_df.to_string())
    
    # 8. Prepare predictions dataframe
    test_data_copy = test_data.copy()
    test_data_copy['new_ys_prediction'] = predictions.flatten()
    test_data_copy['model_train_date'] = model_date
    
    # 9. Upload to BigQuery if requested
    if upload_to_bq and len(test_data_copy) > 0:
        print(f"\nUploading {len(test_data_copy)} predictions to BigQuery...")
        
        # Prepare data for upload - include additional columns
        upload_data = test_data_copy[['rtrs_control_number', 'cusip', 'trade_date', 
                                     'dollar_price', 'yield', 'new_ficc_ycl', 
                                     'new_ys', 'new_ys_prediction', 
                                     'trade_datetime', 'trade_type']].copy()
        
        # Add quantity (in thousands) - either actual or fixed
        if fixed_quantity:
            upload_data['quantity'] = fixed_quantity  # Fixed quantity in thousands
            upload_data['quantity_override'] = True
        else:
            # Convert from log10 back to thousands
            upload_data['quantity'] = np.round(10 ** test_data_copy['quantity'] / 1000, 2)
            upload_data['quantity_override'] = False
        
        # Add trade_type_override flag
        upload_data['trade_type_override'] = cycle_trade_types
        
        # Calculate error in basis points
        # Error = actual yield - (new_ficc_ycl + new_ys_prediction)
        upload_data['error_bps'] = (upload_data['yield'] - 
                                    (upload_data['new_ficc_ycl'] + upload_data['new_ys_prediction']))
        
        EASTERN = timezone('US/Eastern')
        upload_data['prediction_datetime'] = pd.to_datetime(datetime.now(EASTERN).replace(microsecond=0))
        upload_data['trade_date'] = pd.to_datetime(upload_data['trade_date']).dt.date
        
        # Add extra fields with correct types
        upload_data['model_train_date'] = pd.to_datetime(model_date).date()
        if fixed_quantity:
            if cycle_trade_types:
                upload_data['evaluation_mode'] = 'fixed_100k_all_trade_types'
            else:
                upload_data['evaluation_mode'] = 'fixed_quantity_100k'
        else:
            upload_data['evaluation_mode'] = 'historical'
        upload_data['days_since_training'] = (pd.to_datetime(test_date) - pd.to_datetime(model_date)).days
        out = upload_data[[
            'rtrs_control_number',
            'cusip',
            'trade_date',
            'dollar_price',
            'yield',
            'new_ficc_ycl',
            'new_ys',
            'new_ys_prediction',
            'prediction_datetime',
        ]].copy()
        
        try:
            # Use the upload function from auxiliary_functions
            # from auxiliary_functions import upload_predictions
            
            # Upload
            aux.upload_predictions(out, 'yield_spread_with_similar_trades')
            print(f"Successfully uploaded predictions for {test_date} using model {model_date}")
            
        except Exception as e:
            print(f"Failed to upload to BigQuery: {e}")
    
    return {
        'model_date': model_date,
        'test_date': test_date,
        'result_df': result_df,
        'predictions': test_data_copy,
        'mae': result_df.loc['Entire set', 'Mean Absolute Error'],
        'rmse': result_df.loc['Entire set', 'RMSE'] if 'RMSE' in result_df.columns else None
    }

def evaluate_latest_model_for_all_dates(data: pd.DataFrame = None, upload_to_bq: bool = True, 
                                        start_date: str = None, end_date: str = None,
                                        fixed_quantity: int = None, cycle_trade_types: bool = False):
    """
    For each date in the data, use the most recent model available to make predictions.
    This simulates production where we use the latest model until a new one is trained.
    
    Args:
        data: DataFrame with trade data
        upload_to_bq: Whether to upload results to BigQuery
        start_date: Start date for evaluation (YYYY-MM-DD)
        end_date: End date for evaluation (YYYY-MM-DD)
        fixed_quantity: If provided, use this fixed quantity (in thousands) for all trades
        cycle_trade_types: If True, create predictions for all trade types (D, S, P)
    """
    import pandas as pd
    from datetime import datetime
    
    # Get all available models
    models = list_similar_trades_models()
    model_dates = [m['date'] for m in models]
    model_dates.sort()  # Ensure chronological order
    
    # Load data if not provided
    if data is None:
        data = get_processed_data_pickle_file(MODEL)
    
    # Get all available test dates
    all_dates = sorted(data['trade_date'].unique())
    
    # Filter dates if range specified
    if start_date:
        all_dates = [d for d in all_dates if str(d) >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if str(d) <= end_date]
    
    results = []
    
    # For each date, find the most recent model
    for test_date in all_dates:
        # Find the most recent model before this date
        test_date_str = str(pd.to_datetime(test_date).date())
        available_models = [m for m in model_dates if m < test_date_str]
        
        if not available_models:
            print(f"No model available for {test_date}")
            continue
        
        model_date = available_models[-1]  # Most recent model
        
        try:
            print(f"Testing {test_date} using model from {model_date}...")
            result = evaluate_model_on_single_day(
                model_date, 
                test_date, 
                data=data,
                upload_to_bq=upload_to_bq,
                fixed_quantity=fixed_quantity,
                cycle_trade_types=cycle_trade_types
            )
            if result:
                results.append(result)
                days_old = (pd.to_datetime(test_date) - pd.to_datetime(model_date)).days
                print(f"  ✓ MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}, Model age: {days_old} days")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'model_date': r['model_date'],
            'test_date': r['test_date'],
            'days_since_training': (pd.to_datetime(r['test_date']) - pd.to_datetime(r['model_date'])).days,
            'mae': r['mae'],
            'rmse': r['rmse'],
            'ig_mae': r['result_df'].loc['Investment Grade', 'Mean Absolute Error'],
            'ig_rmse': r['result_df'].loc['Investment Grade', 'RMSE'] if 'RMSE' in r['result_df'].columns else None,
            'num_trades': int(r['result_df'].loc['Entire set', 'Trade Count'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("\nNo results to summarize - check date ranges and available models")
        return summary_df, pd.DataFrame()
    
    # Add some analysis
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Average performance by days since training
    days_perf = summary_df.groupby('days_since_training').agg({
        'mae': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance by Days Since Training:")
    print(days_perf)
    
    # Save complete results with descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if fixed_quantity:
        suffix = f"_fixed{fixed_quantity}k"
        if cycle_trade_types:
            suffix += "_DSP"
    
    summary_file = f'production_evaluation_summary{suffix}_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved detailed results to {summary_file}")
    
    # Also save aggregated results
    model_summary = summary_df.groupby('model_date').agg({
        'mae': ['mean', 'std', 'min', 'max', 'count'],
        'rmse': ['mean', 'std', 'min', 'max'],
        'days_since_training': 'max'
    }).round(2)
    
    model_summary_file = f'model_performance_summary{suffix}_{timestamp}.csv'
    model_summary.to_csv(model_summary_file)
    print(f"Saved model summary to {model_summary_file}")
    
    return summary_df, model_summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or evaluate yield spread models')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'list', 'production-eval'], default='train',
                        help='Mode: train new models, evaluate existing models, list available models, or run production evaluation')
    parser.add_argument('--model-dates', nargs='+', help='Model dates to evaluate (YYYY-MM-DD)')
    parser.add_argument('--test-dates', nargs='+', help='Test dates to evaluate on (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for production-eval (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for production-eval (YYYY-MM-DD)')
    parser.add_argument('--fixed-quantity', type=int, default=None, 
                        help='Use fixed quantity (in thousands) for all trades. E.g., 100 means 100k par value')
    parser.add_argument('--cycle-trade-types', action='store_true',
                        help='Create predictions for all trade types (D, S, P) for each trade')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        list_similar_trades_models()
    
    elif args.mode == 'production-eval':
        # Run the comprehensive production evaluation
        aux.setup_gpus(False)
        print("Starting comprehensive production evaluation...")
        
        if args.start_date or args.end_date:
            print(f"Date range: {args.start_date or 'beginning'} to {args.end_date or 'end'}")
        
        if args.fixed_quantity:
            print(f"Using fixed quantity of {args.fixed_quantity}k for all trades")
            if args.cycle_trade_types:
                print("Creating predictions for all trade types (D, S, P)")
        
        print("This will test each model on all days it was in production.")
        print("This may take several hours to complete.\n")
        
        # Update evaluate_latest_model_for_all_dates to pass through the fixed quantity params
        summary_df, model_summary = evaluate_latest_model_for_all_dates(
            upload_to_bq=True, 
            start_date=args.start_date,
            end_date=args.end_date,
            fixed_quantity=args.fixed_quantity,
            cycle_trade_types=args.cycle_trade_types
        )
        
        print("\n\nEvaluation complete!")
        print(f"Tested {len(summary_df)} model-date combinations")
        print(f"Results uploaded to BigQuery sandbox table")
        
    elif args.mode == 'evaluate':
        # Example: python train_model.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04
        if not args.model_dates or not args.test_dates:
            print("Please provide --model-dates and --test-dates for evaluation")
            sys.exit(1)
        
        aux.setup_gpus(False)
        data = get_processed_data_pickle_file(model = MODEL)
        
        # For single evaluations, just loop through the combinations
        for model_date in args.model_dates:
            for test_date in args.test_dates:
                print(f"\nEvaluating model {model_date} on {test_date}")
                result = evaluate_model_on_single_day(
                    model_date, 
                    test_date, 
                    data=data, 
                    upload_to_bq=True,
                    fixed_quantity=args.fixed_quantity,
                    cycle_trade_types=args.cycle_trade_types
                )
                if result:
                    print(f"MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
   

    else:  # train mode (default)
        aux.setup_gpus(False)
        #Pass in 'date' here to enforce a end date for the sliding window of models other than the most recent date in the data
        data = get_processed_data_pickle_file(model = MODEL)
        output_file_name = 'output.txt'
        #Change num_days to train models over a larger sliding window of dates
        train_model_from_data_file(data, NUM_DAYS, output_file_name) 
