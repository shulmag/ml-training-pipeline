from functools import wraps
import warnings
from datetime import datetime, date, time

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates


warnings.simplefilter(action='ignore', category=FutureWarning)


YEAR_MONTH_DAY_HOUR_MIN = '%Y-%m-%d:%H:%M'
YEAR_MONTH_DAY_HOUR_MIN_SEC = '%Y-%m-%d:%H:%M:%S'
START_DATE = '2021-08-03'    # '2021-07-27' was for the daily yield curve, but '2021-08-03' is for the realtime yield curve
START_TIME = '09:30'


MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(15, 59)


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

BUSINESS_DAY = CustomBusinessDay(calendar=USHolidayCalendarWithGoodFriday())    # used to skip over holidays when adding or subtracting business days


def cache(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        # creating a key using maturity and target datetime
        cache_key = str(args[0]) + str(args[1])
        if cache_key in wrapper.cache:
            output = wrapper.cache[cache_key]
        else:
            output = function(*args, **kwargs)
            wrapper.cache[cache_key] = output
        return output

    wrapper.cache = dict()
    return wrapper


def get_duration(row, use_last_calc_day_cat: bool) -> float:
    '''Get the duration in years from `row`. If `use_last_calc_day_cat` is `True`, then the function will 
    use the last duration for the yield curve level. Otherwise, it will use the current duration by 
    using the calculation date computed upstream.

    NOTE: there is an argument the logic when setting `use_last_calc_day_cat` to `False` causes leakage 
    because in production we do not have the `calc_date` column. However, the objective is to train a 
    model that performs as well as possible IF we had the correct calc date. In production, we have a 
    subprocedure to infer what the calc date is, so it makes sense to train a model that performs as well 
    as possible if the calc date were to be correct. Subpoint: in most cases, the calc date is known (e.g., 
    for bond without a call option, a called bond, etc.).'''
    start_date = row['trade_date']    # this value is never null (verified from inspecting the materialized trade history on 2025-05-13); previously used `settlement_date` here but there are some discrepancies with the `trade_date` especially for new issues (see Jira task: https://ficcai.atlassian.net/browse/FA-2980)
    
    maturity_date = row['maturity_date']    # this value is never null (verified from inspecting the materialized trade history on 2025-05-13)
    if use_last_calc_day_cat:
        # the reason that we need to use the logic below instead of just using row['last_calc_date'] is because 
        # (1) if the bond has been called, then the last calc date is incorrect
        # (2) if the reference data is updated, then using the category and selecting the correct date is more accurate since last_calc_date may be a date in the past since the next_call_date has been updated
        is_called, is_callable, last_calc_day_cat = row['is_called'], row['is_callable'], row['last_calc_day_cat']
        if is_called is True:
            end_date = row['refund_date']    # this value is never null if `is_called` is True (verified from inspecting the materialized trade history on 2025-05-13)
        elif is_callable is False:
            end_date = maturity_date
        elif last_calc_day_cat == 0 and pd.notna(row['next_call_date']):    # sometimes the case that `next_call_date` is null, but `first_call_date` is not null, and in this case, the correct value is perhaps `first_call_date` but requires complicated upstream code to correct it and deeper investigation before making the change here (1.3% of materialized trade history on 2025-05-13, and only new issues)
            end_date = row['next_call_date']
        elif last_calc_day_cat == 1 and pd.notna(row['par_call_date']):
            end_date = row['par_call_date']
        else:
            end_date = maturity_date
    else:    # `use_last_calc_day_cat` is `False`
        calc_date = row['calc_date']
        if pd.isna(calc_date):
            end_date = maturity_date
        else:
            end_date = calc_date

    return diff_in_days_two_dates(end_date, start_date) / NUM_OF_DAYS_IN_YEAR


def get_yield_curve_level(row, nelson_params: dict, scalar_params: dict, shape_params: dict, end_of_day: bool, use_last_calc_day_cat: bool) -> float:
    '''`end_of_day` is a boolean that indicates whether the data is end-of-day yield curve or the real-time (minute) yield curve. 
    `use_last_calc_day_cat` is a boolean that indicates whether to use the last duration for the yield curve level. If `use_last_calc_day_cat` 
    is `True`, then the function will use the last duration for the yield curve level. Otherwise, it will use the current duration by 
    using the calculation date computed upstream.'''
    return yield_curve_level(get_duration(row, use_last_calc_day_cat), row['trade_datetime'], nelson_params, scalar_params, shape_params, end_of_day)



def decay_transformation(t: np.array, L: float):
    '''Takes a numpy array of maturities (or a single float) and a shape parameter, and returns the exponential function
    calculated from those values. This is the first feature of the Nelson-Siegel model.'''
    return L * (1 - np.exp(-t / L)) / t


def laguerre_transformation(t: np.array, L: float):
    '''Takes a numpy array of maturities (or a single float) and a shape parameter, and returns the laguerre function
    calculated from those values. This is the second feature of the Nelson-Siegel model.'''
    return (L * (1 - np.exp(-t / L)) / t) - np.exp(-t / L)


def is_time_between(begin_time, end_time, check_time):
    '''If check time is not given, default to current UTC time.'''
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else:    # crosses midnight
        return check_time >= begin_time or check_time <= end_time


def iterate_backward_by_minute(dictionary: dict, most_recent_datetime: datetime) -> datetime:
    '''Checks whether the datetime exists as a key in `dictionary`. If not, we loop back to the previous datetime key. 
    NOTE: this is taken directly from `modules/ficc/utils/yc_data.py::find_last_minute(...)` with minor modifications 
    such as checking a dictionary instead of a redis.'''
    def iterate_backward(date_and_time, num_mins_back, max_num_times=-1):
        '''Iterate backward from `date_and_time` in chunks of `num_mins_back` for a maximum of 
        `max_num_times`. If the date and time are not found, i.e., the `max_num_times` has been 
        reached, then return the last searched date and time, with the second argument as `False`. 
        Otherwise, return the second argument as `True`, with the date and time whenever it is found.'''
        while date_and_time not in dictionary:    # find the `date_and_time` that is at most `num_mins_back` minutes in the past
            if max_num_times == 0: return date_and_time, False
            if is_time_between(MARKET_OPEN, MARKET_CLOSE, date_and_time.time()):    # only in between 9:30 am ET (market open) to 4pm ET
                date_and_time -= pd.Timedelta(minutes=num_mins_back)
            else:
                if date_and_time.time() < MARKET_OPEN:    # if the time is before 9:30am ET (market open), then we need to go to the previous business day
                    date_and_time = date_and_time - (BUSINESS_DAY * 1)
                date_and_time = date_and_time.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute)
            max_num_times -= 1
        return date_and_time, True
    
    def iterate_forward(date_and_time, num_mins_forward):
        '''Iterate forward from `date_and_time` until the date and time is no longer found in the 
        redis. Once it is not found, return the previous date and time inspected to be the most 
        recent date and time that was found in the redis.'''
        while date_and_time in dictionary:    # after finding `date_and_time` to most `num_mins_back` minutes in the past, increment up by 1 minute until we cannot find it anymore
            date_and_time += pd.Timedelta(minutes=num_mins_forward)
        return date_and_time - pd.Timedelta(minutes=num_mins_forward)
    
    def iterate_backward_backward_forward(date_and_time, num_mins):
        '''Iterate back minute by minute for the first `num_mins` minutes. If the date and time does not 
        exist in the redis client, then start iterating back by `num_mins` chunks. Once the date and time 
        is found, iterate forward minute by minute until date and time is no longer found. This will be 
        the most recent date and time for which we have coefficients.'''
        date_and_time, found = iterate_backward(date_and_time, 1, num_mins)
        if found: return date_and_time
        date_and_time, _ = iterate_backward(date_and_time, num_mins)
        return iterate_forward(date_and_time, 1)

    most_recent_datetime = most_recent_datetime.replace(second=0, microsecond=0)    # remove seconds and milliseconds since downstream comparisons do not use them
    start_date_time = datetime.strptime(START_DATE + ':' + START_TIME, YEAR_MONTH_DAY_HOUR_MIN)
    if most_recent_datetime <= start_date_time:
        return start_date_time
    else:
        return iterate_backward_backward_forward(most_recent_datetime, 5)


def iterate_backward_by_day(dictionary: dict, most_recent_date: date) -> date:
    '''Iterates backwards through the dictionary until it finds the most recent date that is in the dictionary at or before `most_recent_date`.'''
    keys = set(dictionary.keys())
    while most_recent_date not in keys:
        most_recent_date = (most_recent_date - (BUSINESS_DAY * 1)).date()
    return most_recent_date


def load_model_parameters(target_datetime: datetime, nelson_params: dict, scalar_params: dict, shape_parameter: dict, end_of_day: bool):
    '''Grabs the Nelson-Siegel coefficients and standard scalar coefficient from the input dataframes. 
    `end_of_day` is a boolean that indicates whether the data is end-of-day yield curve or the real-time (minute) yield curve.'''
    previous_date = (target_datetime - (BUSINESS_DAY * 1)).date()
    if end_of_day:
        target_datetime_for_nelson_params = iterate_backward_by_day(nelson_params, previous_date)
    else:    # if the data is real-time, we need to find the most recent minute
        target_datetime_for_nelson_params = iterate_backward_by_minute(nelson_params, target_datetime)
    nelson_coeff = list(nelson_params[target_datetime_for_nelson_params].values()) + [target_datetime_for_nelson_params]

    target_date_for_scalar_params = iterate_backward_by_day(scalar_params, previous_date)
    scalar_coeff = list(scalar_params[target_date_for_scalar_params].values()) + [target_date_for_scalar_params]

    target_date_for_shape_parameter = iterate_backward_by_day(shape_parameter, previous_date)
    shape_param = (shape_parameter[target_date_for_shape_parameter]['L'], target_date_for_shape_parameter)

    return nelson_coeff, scalar_coeff, shape_param


def get_scaled_features(t: np.array, exponential_mean: float, exponential_std: float, laguerre_mean: float, laguerre_std: float, shape_paramter: float):
    '''Takes as input the parameters loaded from the scaler parameter table in bigquery on a given day, alongside an array (or a
    single float) value to be scaled as input to make predictions. It then manually recreate the transformations from the sklearn
    StandardScaler used to scale data in training by first creating the exponential and laguerre functions then scaling them.'''
    X1 = (decay_transformation(t, shape_paramter) - exponential_mean) / exponential_std
    X2 = (laguerre_transformation(t, shape_paramter) - laguerre_mean) / laguerre_std
    return X1, X2


def predict_yield_curve_level(maturity: np.array,
                              const: float,
                              exponential: float,
                              laguerre: float,
                              exponential_mean: float,
                              exponential_std: float,
                              laguerre_mean: float,
                              laguerre_std: float,
                              shape_parameter: float):
    '''Wrapper function that takes the prediction inputs, the scaler parameters and the model parameters from a given day. It then
    scales the input using the get_scaled_features function to obtain the model inputs, and predicts the yield-to-worst implied by the
    nelson-siegel model on that day. Because the Nelson-Siegel model is linear, we can do a simple calculation.'''
    X1, X2 = get_scaled_features(maturity, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_parameter)
    return const + exponential * X1 + laguerre * X2


@cache
def yield_curve_level(maturity: float, target_datetime, nelson_params, scalar_params, shape_params, end_of_day: bool):
    '''`maturity` is the gap in years between the yield-to-worst date and the target date from which we want the yield 
    curve used in the ytw calculations to be from. `end_of_day` is a boolean that indicates whether the data is end-of-day 
    yield curve or the real-time (minute) yield curve.'''
    nelson_siegel_coef, scaler_daily_parameters, shape_parameter = load_model_parameters(target_datetime, nelson_params, scalar_params, shape_params, end_of_day)
    const, exponential, laguerre, target_datetime_for_nelson_params = nelson_siegel_coef
    exponential_mean, exponential_std, laguerre_mean, laguerre_std, target_date_for_scaler_params = scaler_daily_parameters
    shape_parameter, target_date_for_shape_parameter = shape_parameter
    ycl_prediction = predict_yield_curve_level(maturity, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_parameter)
    return ycl_prediction, maturity, const, exponential, laguerre, target_datetime_for_nelson_params, exponential_mean, exponential_std, laguerre_mean, laguerre_std, target_date_for_scaler_params, shape_parameter, target_date_for_shape_parameter
