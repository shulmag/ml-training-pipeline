'''
Description: Framework for evaluating dollar price models.

This script evaluates dollar price models by:
- Loading historical dollar price models from GCS
- Testing them on specific dates
- Uploading predictions to BigQuery for analysis
- Supporting fixed quantity (100k) evaluation with trade type cycling

USAGE MODES:
1. list: Show all available dollar price models in GCS
   $ python evaluate_dollar_price.py --mode list

2. evaluate: Test specific model(s) on specific date(s)
   $ python evaluate_dollar_price.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04

3. production-eval: Comprehensive backtesting
   $ python evaluate_dollar_price.py --mode production-eval --start-date 2025-04-21 --end-date 2025-07-21

BACKGROUND EXECUTION:
$ nohup python -u evaluate_dollar_price.py --mode production-eval >> output_dp.txt 2>&1 &
'''

import os
import sys
import pickle
import numpy as np
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
import re
from datetime import datetime
from pytz import timezone
import gcsfs

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/gil/git/ficc_python/creds.json'

# Add paths
ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'automated_training'))
sys.path.append(ficc_package_dir)

from auxiliary_variables import MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, BUCKET_NAME
import auxiliary_functions
auxiliary_functions.SAVE_MODEL_AND_DATA = True
from auxiliary_functions import setup_gpus, get_optional_arguments_for_process_data
from clean_training_log import remove_lines_with_tensorflow_progress_bar

ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ficc'))
sys.path.append(ficc_package_dir)

from utils.auxiliary_functions import function_timer, get_dp_trade_history_features

# Dollar price model configuration
MODEL = 'dollar_price'
PICKLE_FILE_PATH = "/home/gil/git/ficc_python/notebooks/train_model/processed_data_dollar_price_v2.pkl"

def list_dollar_price_models(bucket_name='automated_training', prefix='dollar-v2-model-'):
    """List all dollar-v2 models in GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    models = []
    seen_dates = set()
    
    for blob in bucket.list_blobs(prefix=prefix):
        # Extract date from model name (dollar-v2-model-YYYY-MM-DD/)
        match = re.search(r'dollar-v2-model-(\d{4}-\d{2}-\d{2})', blob.name)
        if match:
            date_str = match.group(1)
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                models.append({
                    'date': date_str,
                    'path': f'gs://{bucket_name}/dollar-v2-model-{date_str}/',
                    'name': f'dollar-v2-model-{date_str}'
                })
    
    models.sort(key=lambda x: x['date'], reverse=True)
    
    print(f"Found {len(models)} dollar-v2 models:")
    for model in models[:10]:
        print(f"  {model['date']}: {model['path']}")
    
    return models

@function_timer
def get_processed_data_pickle_file():
    """Load the dollar price processed data pickle file"""
    print(f'Loading data from {PICKLE_FILE_PATH}...')
    with open(PICKLE_FILE_PATH, 'rb') as file:
        data = pickle.load(file)
    most_recent_trade_datetime = data.trade_datetime.max()
    print(f'Loaded data. Most recent trade datetime: {most_recent_trade_datetime}')
    return data

def get_num_features_for_each_trade_in_history():
    """Get the number of features for dollar price model"""
    optional_arguments = get_optional_arguments_for_process_data(MODEL)
    trade_history_features = get_dp_trade_history_features()
    return len(trade_history_features)

def target_trade_processing_for_attention(row):
    """Create target attention features for dollar price model"""
    trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
    return np.tile(target_trade_features, (1, 1))

def create_dollar_price_bq_table(bq_client, table_id):
    """Create BigQuery table for dollar price predictions if it doesn't exist"""
    schema = [
        bigquery.SchemaField('rtrs_control_number', 'INTEGER', 'REQUIRED'),
        bigquery.SchemaField('cusip', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('trade_date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('dollar_price', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('dollar_price_prediction', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('trade_datetime', 'DATETIME', 'REQUIRED'),
        bigquery.SchemaField('trade_type', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('quantity', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('quantity_override', 'BOOLEAN', 'REQUIRED'),
        bigquery.SchemaField('trade_type_override', 'BOOLEAN', 'REQUIRED'),
        bigquery.SchemaField('error_dollars', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('prediction_datetime', 'DATETIME', 'REQUIRED'),
        bigquery.SchemaField('model_train_date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('evaluation_mode', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('days_since_training', 'INTEGER', 'REQUIRED')
    ]
    
    try:
        bq_client.get_table(table_id)
        print(f"Table {table_id} already exists")
    except:
        print(f"Creating table {table_id}")
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)

def evaluate_dollar_price_model_on_single_day(
    model_date: str,
    test_date: str,
    data: pd.DataFrame = None,
    upload_to_bq: bool = True,
    fixed_quantity: float = None,
    cycle_trade_types: bool = False
):
    """Load a pre-trained dollar price model and evaluate it on a specific test date"""
    
    import tensorflow as tf
    from tensorflow import keras
    
    # 1. Load the data if not provided
    if data is None:
        data = get_processed_data_pickle_file()
    
    # 2. Filter to test date
    test_data = data[data['trade_date'] == test_date]
    print(f"Test data shape for {test_date}: {test_data.shape}")
    
    if len(test_data) == 0:
        print(f"No data found for test date {test_date}")
        return None
    
    # 3. Handle fixed quantity and trade type cycling
    if fixed_quantity is not None:
        if cycle_trade_types:
            # Create 3 copies of each trade, one for each trade type
            test_data_expanded = []
            trade_types = ['D', 'S', 'P']
            
            for _, row in test_data.iterrows():
                for trade_type in trade_types:
                    row_copy = row.copy()
                    row_copy['quantity'] = np.log10(fixed_quantity * 1000)
                    row_copy['par_traded'] = fixed_quantity * 1000
                    row_copy['trade_type'] = trade_type
                    
                    # Update target_attention_features
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
            
            # Update target_attention_features
            for idx in test_data.index:
                trade_type = test_data.loc[idx, 'trade_type']
                trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
                target_features = [test_data.loc[idx, 'quantity']] + trade_mapping[trade_type]
                test_data.loc[idx, 'target_attention_features'] = np.tile(target_features, (1, 1))
    
    # 4. Load the model
    model_path = f'gs://automated_training/dollar-v2-model-{model_date}'
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # 5. Load the encoders
    fs = gcsfs.GCSFileSystem()
    encoders_path = 'gs://automated_training/encoders_dollar_price.pkl'
    with fs.open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    print(f"Loaded encoders from {encoders_path}")
    
    # 6. Create inputs for dollar price model
    from auxiliary_functions import create_input
    x_test, y_test = create_input(test_data, encoders, 'dollar_price')
    
    # 7. Generate predictions
    predictions = model.predict(x_test, batch_size=1000)
    
    # 8. Calculate metrics
    from auxiliary_functions import segment_results
    delta = np.abs(predictions.flatten() - y_test)
    result_df = segment_results(test_data, delta)
    
    # Add RMSE calculation
    rmse = np.sqrt(np.mean((predictions.flatten() - y_test) ** 2))
    result_df['RMSE'] = rmse
    
    print(f"\nResults for dollar price model {model_date} on test date {test_date}:")
    if fixed_quantity:
        if cycle_trade_types:
            print(f"(Using fixed quantity of {fixed_quantity}k and cycling through trade types D, S, P)")
        else:
            print(f"(Using fixed quantity of {fixed_quantity}k)")
    print(result_df.to_string())
    
    # 9. Prepare predictions dataframe
    test_data_copy = test_data.copy()
    test_data_copy['dollar_price_prediction'] = predictions.flatten()
    test_data_copy['model_train_date'] = model_date
    
    # 10. Upload to BigQuery if requested
    if upload_to_bq and len(test_data_copy) > 0:
        print(f"\nUploading {len(test_data_copy)} predictions to BigQuery...")
        
        # Prepare data for upload
        upload_data = test_data_copy[['rtrs_control_number', 'cusip', 'trade_date', 
                                     'dollar_price', 'trade_datetime', 'trade_type']].copy()
        
        upload_data['dollar_price_prediction'] = test_data_copy['dollar_price_prediction']
        
        # Add quantity (in thousands)
        if fixed_quantity:
            upload_data['quantity'] = fixed_quantity
            upload_data['quantity_override'] = True
        else:
            upload_data['quantity'] = np.round(10 ** test_data_copy['quantity'] / 1000, 2)
            upload_data['quantity_override'] = False
        
        upload_data['trade_type_override'] = cycle_trade_types
        
        # Calculate error in dollars
        upload_data['error_dollars'] = upload_data['dollar_price'] - upload_data['dollar_price_prediction']
        
        EASTERN = timezone('US/Eastern')
        upload_data['prediction_datetime'] = pd.to_datetime(datetime.now(EASTERN).replace(microsecond=0))
        upload_data['trade_date'] = pd.to_datetime(upload_data['trade_date']).dt.date
        
        # Add metadata
        upload_data['model_train_date'] = pd.to_datetime(model_date).date()
        if fixed_quantity:
            if cycle_trade_types:
                upload_data['evaluation_mode'] = 'fixed_100k_all_trade_types'
            else:
                upload_data['evaluation_mode'] = 'fixed_quantity_100k'
        else:
            upload_data['evaluation_mode'] = 'historical'
        upload_data['days_since_training'] = (pd.to_datetime(test_date) - pd.to_datetime(model_date)).days
        
        try:
            # Create table if needed
            from google.cloud import bigquery
            bq_client = bigquery.Client()
            table_id = 'eng-reactor-287421.historic_predictions.historical_predictions_dollar_price_v2'
            
            # Upload
            job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')
            job = bq_client.load_table_from_dataframe(upload_data, table_id, job_config=job_config)
            job.result()
            
            print(f"Successfully uploaded predictions for {test_date} using model {model_date}")
            
        except Exception as e:
            print(f"Failed to upload to BigQuery: {e}")
    
    return {
        'model_date': model_date,
        'test_date': test_date,
        'result_df': result_df,
        'predictions': test_data_copy,
        'mae': result_df.loc['Entire set', 'Mean Absolute Error'],
        'rmse': rmse
    }

def evaluate_latest_dollar_model_for_all_dates(
    data: pd.DataFrame = None, 
    upload_to_bq: bool = True,
    start_date: str = None, 
    end_date: str = None,
    fixed_quantity: int = None, 
    cycle_trade_types: bool = False
):
    """
    For each date, use the most recent dollar price model available to make predictions.
    """
    # Get all available models
    models = list_dollar_price_models()
    model_dates = [m['date'] for m in models]
    model_dates.sort()
    
    # Load data if not provided
    if data is None:
        data = get_processed_data_pickle_file()
    
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
            result = evaluate_dollar_price_model_on_single_day(
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
            'ig_mae': r['result_df'].loc['Investment Grade', 'Mean Absolute Error'] if 'Investment Grade' in r['result_df'].index else None,
            'num_trades': int(r['result_df'].loc['Entire set', 'Trade Count'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("\nNo results to summarize - check date ranges and available models")
        return summary_df, pd.DataFrame()
    
    # Analysis
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS - DOLLAR PRICE MODEL")
    print(f"{'='*80}")
    
    # Average performance by days since training
    days_perf = summary_df.groupby('days_since_training').agg({
        'mae': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance by Days Since Training:")
    print(days_perf)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if fixed_quantity:
        suffix = f"_fixed{fixed_quantity}k"
        if cycle_trade_types:
            suffix += "_DSP"
    
    summary_file = f'dollar_price_evaluation_summary{suffix}_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved detailed results to {summary_file}")
    
    # Model summary
    model_summary = summary_df.groupby('model_date').agg({
        'mae': ['mean', 'std', 'min', 'max', 'count'],
        'rmse': ['mean', 'std', 'min', 'max'],
        'days_since_training': 'max'
    }).round(2)
    
    model_summary_file = f'dollar_price_model_performance_summary{suffix}_{timestamp}.csv'
    model_summary.to_csv(model_summary_file)
    print(f"Saved model summary to {model_summary_file}")
    
    return summary_df, model_summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate dollar price models')
    parser.add_argument('--mode', choices=['list', 'evaluate', 'production-eval'], default='list',
                        help='Mode: list available models, evaluate specific models, or run production evaluation')
    parser.add_argument('--model-dates', nargs='+', help='Model dates to evaluate (YYYY-MM-DD)')
    parser.add_argument('--test-dates', nargs='+', help='Test dates to evaluate on (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for production-eval (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for production-eval (YYYY-MM-DD)')
    parser.add_argument('--fixed-quantity', type=int, default=None,
                        help='Use fixed quantity (in thousands) for all trades')
    parser.add_argument('--cycle-trade-types', action='store_true',
                        help='Create predictions for all trade types (D, S, P) for each trade')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        list_dollar_price_models()
    
    elif args.mode == 'production-eval':
        setup_gpus(False)
        print("Starting dollar price model production evaluation...")
        
        if args.start_date or args.end_date:
            print(f"Date range: {args.start_date or 'beginning'} to {args.end_date or 'end'}")
        
        if args.fixed_quantity:
            print(f"Using fixed quantity of {args.fixed_quantity}k for all trades")
            if args.cycle_trade_types:
                print("Creating predictions for all trade types (D, S, P)")
        
        summary_df, model_summary = evaluate_latest_dollar_model_for_all_dates(
            upload_to_bq=True,
            start_date=args.start_date,
            end_date=args.end_date,
            fixed_quantity=args.fixed_quantity,
            cycle_trade_types=args.cycle_trade_types
        )
        
        print("\n\nEvaluation complete!")
        print(f"Tested {len(summary_df)} model-date combinations")
        print(f"Results uploaded to BigQuery sandbox table: historical_predictions_dollar_price_v2")
    
    elif args.mode == 'evaluate':
        if not args.model_dates or not args.test_dates:
            print("Please provide --model-dates and --test-dates for evaluation")
            sys.exit(1)
        
        setup_gpus(False)
        data = get_processed_data_pickle_file()
        
        for model_date in args.model_dates:
            for test_date in args.test_dates:
                print(f"\nEvaluating dollar price model {model_date} on {test_date}")
                result = evaluate_dollar_price_model_on_single_day(
                    model_date,
                    test_date,
                    data=data,
                    upload_to_bq=True,
                    fixed_quantity=args.fixed_quantity,
                    cycle_trade_types=args.cycle_trade_types
                )
                if result:
                    print(f"MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")