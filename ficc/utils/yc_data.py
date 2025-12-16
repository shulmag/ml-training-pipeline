import pandas as pd


from ficc.utils.auxiliary_variables import PROJECT_ID
from ficc.utils.auxiliary_functions import function_timer, sqltodf
from ficc.utils.nelson_siegel_model import get_yield_curve_level
from ficc.utils.initialize_pandarallel import initialize_pandarallel


YIELD_CURVE_DATASET_NAME = 'yield_curves_v2'


def get_parameters(table_name: str, bq_client, date_column_name: str = 'date') -> dict:
    '''Return the parameters from `table_name` as a dictionary.'''
    params = sqltodf(f'SELECT * FROM `{PROJECT_ID}.{YIELD_CURVE_DATASET_NAME}.{table_name}` ORDER BY {date_column_name} DESC', bq_client)
    params.set_index(date_column_name, drop=True, inplace=True)
    params = params[~params.index.duplicated(keep='first')]
    return params.transpose().to_dict()


@function_timer
def add_yield_curve(data, bq_client, end_of_day: bool = False, use_last_calc_day_cat: bool = False) -> pd.DataFrame:
    '''Add 'new_ficc_ycl' field to `data`. `end_of_day` is a boolean that indicates whether the data is end-of-day yield curve or the real-time (minute) yield curve. 
    `use_last_calc_day_cat` is a boolean that indicates whether to use the last duration for the yield curve level. If `use_last_calc_day_cat` is `True`, then the function 
    will use the last duration for the yield curve level. Otherwise, it will use the current duration by using the calculation date computed upstream.'''
    initialize_pandarallel()    # only initialize if needed

    # TODO: extract the minimum date from `data`, and pass it into `get_parameters(...)` to get the parameters for that date +/- a few days; this will speed up the querying and other downstream procedures
    if end_of_day:
        nelson_params = get_parameters('nelson_siegel_coef_daily', bq_client)
    else:
        nelson_params = get_parameters('nelson_siegel_coef_minute', bq_client)

    scalar_daily_params = get_parameters('standardscaler_parameters_daily', bq_client)
    shape_params = get_parameters('shape_parameters', bq_client, 'Date')    # 'Date' is capitalized for this table which is a typo when initially created

    columns_needed_to_compute_ycl = ['calc_date', 'trade_date', 'settlement_date', 'trade_datetime', 'last_calc_day_cat', 'is_called', 'is_callable', 'refund_date', 'next_call_date', 'par_call_date', 'maturity_date']
    columns_received_from_computing_ycl = ['new_ficc_ycl', 'duration_for_ycl', 'const', 'exponential', 'laguerre', 'target_datetime_for_nelson_params', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std', 'target_date_for_scaler_params', 'shape_parameter', 'target_date_for_shape_parameter']
    get_yield_curve_level_caller = lambda row: get_yield_curve_level(row, nelson_params, scalar_daily_params, shape_params, end_of_day, use_last_calc_day_cat)
    data[columns_received_from_computing_ycl] = data[columns_needed_to_compute_ycl].parallel_apply(get_yield_curve_level_caller, axis=1, result_type='expand')
    return data
