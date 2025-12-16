import io

import pandas as pd

import gcsfs
from google.cloud import storage

from ficc.utils.auxiliary_variables import PROJECT_ID
from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.gcp_storage_functions import download_data


def concatenate_historical_values_and_create_dictionary(values: pd.DataFrame, historical_values: pd.DataFrame, date_column: str = 'date') -> dict:
    """
    Concatenate historical values with the given values DataFrame and create a dictionary from the concatenated DataFrame.

    Args:
        values (pd.DataFrame): The DataFrame containing the current values.
        historical_values (pd.DataFrame): The DataFrame containing the historical values.
        date_column (str): The name of the date column in the DataFrames.

    Returns:
        dict: A dictionary representation of the concatenated DataFrame.
    """
    if historical_values is not None:
        values = pd.concat([values, historical_values])
    values.set_index(date_column, drop=True, inplace=True)
    values = values[~values.index.duplicated(keep='first')]    # Drop rows with duplicate indices, keeping the first such row; this approach was measured fastest per https://stackoverflow.com/a/34297689
    return values.transpose().to_dict()    # Convert to dictionary


def get_historical_yield_curves(use_gcsfs: bool = True) -> pd.DataFrame:
    '''Fetch historical yield curves from Google Cloud Storage. Returns a DataFrame containing the historical yield curves. 
    `use_gcsfs` is a boolean flag that indicates whether to use `gcsfs` for getting the file. If `False`, it gets the file 
    by using the Google Cloud Storage package.'''
    bucket_name = 'automated_training'
    file_name = 'historical_yield_curves.csv'

    if use_gcsfs:    # when running this code locally, `gcsfs` was having problems with certificates
        fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
        with fs.open(f'{bucket_name}/{file_name}') as file:
            historical_yield_curves = pd.read_csv(file)
    else:
        historical_yield_curves = download_data(storage.Client(), 'automated_training', 'historical_yield_curves.csv', deserialize_from_pickle=False)
        if historical_yield_curves is None: raise ValueError(f'{file_name} does not exist in the Google Cloud bucket {bucket_name}.')
        historical_yield_curves = pd.read_csv(io.BytesIO(historical_yield_curves))

    historical_yield_curves.date = pd.to_datetime(historical_yield_curves.date).dt.date
    historical_yield_curves = historical_yield_curves.sort_values('date', ascending=False)
    return historical_yield_curves


def yield_curve_params(bq_client, yield_curve_to_use: str, end_of_day: bool):
    supported_yield_curves = ('FICC', 'FICC_NEW')
    assert yield_curve_to_use in supported_yield_curves, f'Yield curve of {yield_curve_to_use} is not supported. Supported yield curves: {supported_yield_curves}'
    table_name = 'yield_curves' if yield_curve_to_use == 'FICC' else 'yield_curves_v2'

    if end_of_day:
        nelson_siegel_table_name = 'nelson_siegel_coef_daily'
    else:
        nelson_siegel_table_name = 'nelson_siegel_coef_minute'
    nelson_params = sqltodf(f'SELECT * FROM `eng-reactor-287421.{table_name}.{nelson_siegel_table_name}` ORDER BY date DESC', bq_client)
    scalar_params = sqltodf(f'SELECT * FROM `eng-reactor-287421.{table_name}.standardscaler_parameters_daily` ORDER BY date DESC', bq_client)
    shape_parameter = sqltodf('SELECT * FROM `eng-reactor-287421.yield_curves_v2.shape_parameters` ORDER BY Date DESC', bq_client)

    historical_yield_curves = get_historical_yield_curves(False)
    if end_of_day:    # only use historical yield curve values for Nelson-Siegel params for end of day becuase the date values in the index do not have a time component
        temp_nelson_params = historical_yield_curves[['date', 'const', 'exponential', 'laguerre']].copy()
    else:
        temp_nelson_params = None

    temp_scalar_params = historical_yield_curves[['date', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std']].copy()
    temp_shape_parameter = historical_yield_curves[['date', 'L']].copy()
    temp_shape_parameter = temp_shape_parameter.rename(columns={'date': 'Date'})

    nelson_params = concatenate_historical_values_and_create_dictionary(nelson_params, temp_nelson_params)
    scalar_params = concatenate_historical_values_and_create_dictionary(scalar_params, temp_scalar_params)
    shape_parameter = concatenate_historical_values_and_create_dictionary(shape_parameter, temp_shape_parameter, date_column='Date')
    return nelson_params, scalar_params, shape_parameter
