'''
 # @ Create date: 2021-12-17
 # @ Modified date: 2025-01-08
 '''
import warnings

import numpy as np
import pandas as pd


FEATURES_AND_DEFAULT_VALUES = {'purpose_class': 0,    # unknown
                               'call_timing': 0,    # unknown
                               'call_timing_in_part': 0,    # unknown
                               'sink_frequency': 0,    # under special circumstances
                               'sink_amount_type': 10, 
                               'issue_text': 'No issue text', 
                               'state_tax_status': 0, 
                               'series_name': 'No series name', 
                               'transaction_type': 'I', 
                               'next_call_price': 100, 
                               'par_call_price': 100, 
                               'min_amount_outstanding': 0, 
                               'max_amount_outstanding': 0, 
                               'maturity_amount': 0, 
                               'issue_price': lambda df: df.issue_price.mean(skipna=True),    # leakage; computing the mean over the entire dataset uses the test data (low priority, since this barely affects the model)
                               'orig_principal_amount': lambda df: np.log10((10 ** df.orig_principal_amount).mean(skipna=True)),    # leakage; computing the mean over the entire dataset uses the test data (low priority, since this barely affects the model)
                               'par_price': 100, 
                               'called_redemption_type': 0, 
                               'extraordinary_make_whole_call': False, 
                               'make_whole_call': False, 
                               'default_indicator': False, 
                               'called_redemption_type': 0, 
                               'days_to_settle': 0, 
                               'days_to_maturity': 0, 
                               'days_to_refund': 0,    # NOTE: this feature is currently not used for model training, but if we want to use it we should fill in the value appropriately since 0 implies that the bond will be refunded tomorrow even if it has not been called
                               'call_to_maturity': 0,
                               'last_seconds_ago': 0, 
                               'last_yield_spread': 0.0, 
                               'last_dollar_price': 0.0, 
                               'days_in_interest_payment': 180}
FEATURES_AND_DEFAULT_COLUMNS = {'days_to_par': 'days_to_maturity', 
                                'days_to_call': 'days_to_maturity'}


def replace_nan_with_value_or_column(df: pd.DataFrame, feature: str, default_value_or_column, column_or_value: str) -> pd.DataFrame:
    '''`column_or_value` is a string which is either 'column' or 'value' and determines whether `default_value_or_column` should 
    be treated as a column or a value.'''
    assert column_or_value in ('column', 'value'), f'`column_or_value` must be either "column" or "value" but is "{column_or_value}"'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)    # inplace replacements raise `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy return self._update_inplace(result)`
        null_count_for_feature = df[feature].isnull().sum()
        if null_count_for_feature > 0:
            if column_or_value == 'value':
                if callable(default_value_or_column): default_value_or_column = default_value_or_column(df)    # checks whether `default_value_or_column` is a function that needs to be called on the DataFrame
                df[feature] = df[feature].fillna(default_value_or_column)
                replacement_text = f'value {default_value_or_column}'
            else:    # column_or_value == 'column'
                df[feature] = df[feature].fillna(df[default_value_or_column])
                replacement_text = default_value_or_column
            print(f'Filled {feature} with {replacement_text} for {null_count_for_feature} null rows (out of {len(df)} total rows)')
    return df


def fill_missing_values(df):
    for feature, default_value in FEATURES_AND_DEFAULT_VALUES.items():
        try:
            df = replace_nan_with_value_or_column(df, feature, default_value, 'value')
        except Exception as e:
            print(f'{feature} not in dataframe. {type(e)}: {e}')
    for feature, feature_to_replace_it_with in FEATURES_AND_DEFAULT_COLUMNS.items():    # this must be done AFTER replacing with default values since it requires all other columns to have filled in values
        try:
            df = replace_nan_with_value_or_column(df, feature, feature_to_replace_it_with, 'column')
        except Exception as e:
            print(f'{feature} and/or {feature_to_replace_it_with} not in dataframe. {type(e)}: {e}')
    return df
