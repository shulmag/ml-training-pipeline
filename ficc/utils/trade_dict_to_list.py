import numpy as np
import pandas as pd

from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.nelson_siegel_model import yield_curve_level

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def trade_dict_to_list(trade_dict: dict, 
                       remove_short_maturity: bool, 
                       trade_history_delay: int, 
                       use_treasury_spread: bool, 
                       add_rtrs_in_history: bool, 
                       only_dollar_price_history: bool, 
                       yield_curve_to_use: str, 
                       treasury_rate_dict: dict, 
                       nelson_params: dict, 
                       scalar_params: dict, 
                       shape_parameter: dict, 
                       end_of_day: bool) -> list:
    trade_type_mapping = {'D': [0, 0], 'S': [0, 1], 'P': [1, 0]}
    trade_list = []
    necessary_features = [
        'rtrs_control_number',
        'seconds_ago',
        'settlement_date',
        'par_traded',
        'trade_type',
        'seconds_ago',
        'trade_datetime',
        'dollar_price',
    ]
    for feature in necessary_features:
        if trade_dict[feature] is None: return None, None
        if 'date' in feature:
            date_feature_value = trade_dict[feature]
            try:    # sometimes MSRB has errors when entering dates causing a date to be out of bounds or invalid; see Jira task: https://ficcai.atlassian.net/browse/FA-2315
                pd.to_datetime(date_feature_value)
            except Exception as e:
                print(f'For RTRS control number: {trade_dict["rtrs_control_number"]}, when trying to convert {feature} with value: {date_feature_value}, we get the following {type(e)}: {e}')
                return None, None

    if trade_dict['seconds_ago'] < trade_history_delay: return None, None    # only keep trades further in the past than `trade_history_delay`

    trade_datetime = trade_dict['trade_datetime']
    
    if only_dollar_price_history is True:
        yield_at_that_time = None
        yield_spread = None
        trade_list.append(trade_dict['dollar_price'])
    else:
        trade_date = trade_datetime.date()
        if remove_short_maturity is True:
            try:
                days_to_maturity = diff_in_days_two_dates(trade_dict['maturity_date'], trade_date)
                if days_to_maturity < 400:
                    return None, None
            except Exception as e:
                print(f'Failed to remove this trade due to error: {e}')
                for key, value in trade_dict.items():
                    print(f'{key}: {value}')
                return None, None

        calc_date = trade_dict['calc_date']
        time_to_maturity = diff_in_days_two_dates(calc_date, trade_date) / NUM_OF_DAYS_IN_YEAR

        if time_to_maturity <= 0:
            print(f'Skipped the following trade because the time to maturity is nonpositive. RTRS control number: {int(trade_dict["rtrs_control_number"])}\t\tTrade datetime: {trade_datetime}\t\tCalc date: {calc_date}')
            return None, None
        
        if yield_curve_to_use == 'FICC' or yield_curve_to_use == 'FICC_NEW':    # yield curve coefficients are only present before 2021-07-27 for the old yield curve and 2021-08-02 for the new yield curve
            yield_at_that_time = yield_curve_level(time_to_maturity, trade_datetime, nelson_params, scalar_params, shape_parameter, end_of_day)[0]

            if trade_dict['yield'] is not None and yield_at_that_time is not None:
                yield_spread = trade_dict['yield'] * 100 - yield_at_that_time
                trade_list.append(yield_spread)
            else:
                return None, None

        if use_treasury_spread is True:
            treasury_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
            maturity = min(treasury_maturities, key=lambda x: abs(x - time_to_maturity))
            maturity = 'year_' + str(maturity)
            try:
                t_rate = treasury_rate_dict[trade_date][maturity]
            except Exception as e:
                return None, None

            t_spread = (trade_dict['yield'] - t_rate) * 100
            trade_list.append(np.round(t_spread, 3))

    trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))
    trade_list += trade_type_mapping[trade_dict['trade_type']]
    trade_list.append(np.log10(1 + trade_dict['seconds_ago']))

    if add_rtrs_in_history == True:
        trade_list.append(trade_dict['is_non_transaction_based_compensation'])
        trade_list.append(int(trade_dict['rtrs_control_number']))

    trade_features = (
        yield_spread,
        yield_at_that_time,
        int(trade_dict['rtrs_control_number']),
        trade_dict['yield'] * 100 if trade_dict['yield'] is not None else None,
        trade_dict['dollar_price'],
        trade_dict['seconds_ago'],
        float(trade_dict['par_traded']),
        trade_dict['calc_date'],
        trade_dict['maturity_date'],
        trade_dict['next_call_date'],
        trade_dict['par_call_date'],
        trade_dict['refund_date'],
        trade_datetime,
        trade_dict['calc_day_cat'],
        trade_dict['settlement_date'],
        trade_dict['trade_type'],
    )
    return np.stack(trade_list), trade_features
