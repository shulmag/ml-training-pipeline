import warnings

import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import COUPON_FREQUENCY_DICT
from ficc.utils.diff_in_days import diff_in_days
from ficc.utils.days_in_interest_payment import days_in_interest_payment
from ficc.utils.fill_missing_values import fill_missing_values
from ficc.utils.auxiliary_functions import calculate_a_over_e, function_timer


@function_timer
def process_features(df):
    df.interest_payment_frequency.fillna(0, inplace=True)
    df.loc[:, 'interest_payment_frequency'] = df.interest_payment_frequency.apply(lambda x: COUPON_FREQUENCY_DICT[x])
    # TODO: why are some of these features `np.float32` and some are `float`?
    df.loc[:, 'quantity'] = np.log10(df.par_traded.astype(np.float32))
    df.coupon = df.coupon.astype(np.float32)
    df.issue_amount = np.log10(1 + df.issue_amount.astype(np.float32))
    df.maturity_amount = np.log10(1 + df.maturity_amount.astype(float))
    df.orig_principal_amount = np.log10(1 + df.orig_principal_amount.astype(float))
    df.max_amount_outstanding = np.log10(1 + df.max_amount_outstanding.astype(float))
    
    # creating binary features
    df.loc[:, 'callable'] = df.is_callable  
    df.loc[:, 'called'] = df.is_called 
    df.loc[:, 'zerocoupon'] = df.coupon == 0
    df.loc[:, 'whenissued'] = df.delivery_date >= df.trade_date
    df.loc[:, 'sinking'] = ~df.next_sink_date.isnull()
    df.loc[:, 'deferred'] = (df.interest_payment_frequency == 'Unknown') | df.zerocoupon
    
    # converting the dates to a number of days from the settlement date
    # only consider trades to be reportedly correctly if the trades are settled within one month of the trade date
    df.loc[:,'days_to_settle'] = (df.settlement_date - df.trade_date).dt.days.fillna(0)
    num_trades_before_days_to_settle_exclusion = len(df)
    df = df[df.days_to_settle < 30]
    print(f'Removed {num_trades_before_days_to_settle_exclusion - len(df)} trades, since these are settled 30 days or more from trade date')

    # TODO: why do we not do the -np.inf replacement for `call_to_maturity`?
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # ignore `pandas/core/arraylike.py:364: RuntimeWarning: invalid value encountered in log10` since we handle this directly by filling in -np.inf with np.nan
        warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)    # the following np.log10 assignments cause `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead`
        df.loc[:, 'days_to_maturity'] =  np.log10(1 + (df.maturity_date - df.trade_date).dt.days)
        df.loc[:, 'days_to_call'] = np.log10(1 + (df.next_call_date - df.trade_date).dt.days)
        df.loc[:, 'days_to_refund'] = np.log10(1 + (df.refund_date - df.trade_date).dt.days)    # NOTE: this feature is currently not used for model training
        df.loc[:, 'days_to_par'] = np.log10(1 + (df.par_call_date - df.trade_date).dt.days)
        df.loc[:, 'call_to_maturity'] = np.log10(1 + (df.maturity_date - df.next_call_date).dt.days)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)    # inplace replacements raise `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy return self._update_inplace(result)`
        df.days_to_maturity.replace(-np.inf, np.nan, inplace=True)
        df.days_to_call.replace(-np.inf, np.nan, inplace=True)
        df.days_to_refund.replace(-np.inf, np.nan, inplace=True)
        df.days_to_par.replace(-np.inf, np.nan, inplace=True)

        # Adding features from MSRB rule 33G
        df.loc[:, 'accrued_days'] = df.apply(diff_in_days, calc_type='accrual', axis=1)
        df.loc[:, 'days_in_interest_payment'] = df.apply(days_in_interest_payment, axis=1)
        df.loc[:, 'scaled_accrued_days'] = df['accrued_days'] / (360 / df['days_in_interest_payment'])
        df.loc[:, 'A/E'] = df.apply(calculate_a_over_e, axis=1)
    
    df = fill_missing_values(df)
    return df
