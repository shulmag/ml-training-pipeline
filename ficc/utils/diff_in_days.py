'''
 # @ Create date: 2021-12-20
 # @ Modified date: 2024-08-06
 '''
import warnings
import pandas as pd


def _diff_in_days_two_dates_360_30(end_date, start_date):
    '''This function calculates the difference in days using the 360/30 
    convention specified in MSRB Rule Book G-33, rule (e).'''
    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day
    D1 = start_date.day
    D1 = min(D1, 30)
    if D1 == 30: 
        D2 = min(D2, 30)
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)


def _diff_in_days_two_dates_exact(end_date, start_date):
    diff = end_date - start_date
    if isinstance(diff, pd.Series): return diff.dt.days    # https://stackoverflow.com/questions/60879982/attributeerror-timedelta-object-has-no-attribute-dt
    else: return diff.days


ACCEPTED_CONVENTIONS = {'360/30': _diff_in_days_two_dates_360_30, 
                        'exact': _diff_in_days_two_dates_exact}


def diff_in_days_two_dates(end_date, start_date, convention='360/30'):
    if convention not in ACCEPTED_CONVENTIONS:
        print('unknown convention', convention)
        return None
    return ACCEPTED_CONVENTIONS[convention](end_date, start_date)

    
def diff_in_days(trade, convention='360/30', **kwargs):
    # see MSRB Rule 33-G for details
    if 'calc_type' in kwargs:
        calc_type_value = kwargs['calc_type']
        if calc_type_value != 'accrual': raise ValueError(f'"calc_type" is present in kwargs and the only accepted value is "accrual", but the value passed in for "calc_type" is {calc_type_value}')
        if pd.isnull(trade.accrual_date):
            start_date = trade.dated_date
            warnings.warn(f'Since the accrual date is null, we are using the dated date: {start_date}', category=RuntimeWarning)
        else:
            start_date = trade.accrual_date
    else:
        start_date = trade.dated_date    # cannot initialize `start_date` with `trade.dated_date` since it may not exist
    return diff_in_days_two_dates(trade.settlement_date, start_date, convention)
