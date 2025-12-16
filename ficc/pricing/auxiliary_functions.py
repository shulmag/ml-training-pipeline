'''
 # and computing yields.
 '''
import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import compare_dates, dates_are_equal
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.frequency import get_frequency


def transform_reference_data(df):
    '''This function takes the dataframe from the bigquery and updates certain 
    fields to be the right type. Note that this function mutates the fields 
    passed in dataframe, so the function itself has no return value.'''
    df['interest_payment_frequency'] = df.apply(lambda trade: get_frequency(trade['interest_payment_frequency']), axis=1)
    df['coupon'] = df['coupon'].astype(float)
    df['yield'] = df['yield'].astype(float)
    df['deferred'] = (df.interest_payment_frequency == 0) | df.coupon == 0
    
    df['next_call_price'] = df['next_call_price'].astype(float)


def get_next_coupon_date(first_coupon_date, start_date, time_delta):
    '''This function computes the next time a coupon is paid.
    Note that this function could return a `next_coupon_date` that is after the end_date. 
    This does not create a problem since we deal with the final coupon separately in 
    `price_of_bond_with_multiple_periodic_interest_payments`.
    Note that it may be that this function is not necessary because the field 
    `next_coupon_date` is never null when there is a next coupon date. In the 
    future, we should confirm whether this is the case.'''
    date = first_coupon_date
    while compare_dates(date, start_date) < 0:
        date = date + time_delta
    return date
#     cannot use the below code since division is not valid between datetime.timedelta and relativedelta, and converting between types introduces potential for errors
#     num_of_time_periods = int(np.ceil((start_date - first_coupon_date) / time_delta))    # `int` wraps the `ceil` function because the `ceil` function returns a float
#     return first_coupon_date + time_delta * num_of_time_periods


def get_previous_coupon_date(first_coupon_date, start_date, accrual_date, time_delta, next_coupon_date=None):
    '''This function computes the previous time a coupon was paid for this bond 
    by relating it to the next coupon date.
    Note:it may be that this function is not necessary because the field 
    `previous_coupon_date` is never null when `next_coupon_date` exists. In the 
    future, we should confirm whether this is the case.'''
    if next_coupon_date == None:
        next_coupon_date = get_next_coupon_date(first_coupon_date, start_date, time_delta)
        
    if dates_are_equal(next_coupon_date, first_coupon_date):
        return accrual_date
    return next_coupon_date - time_delta


def get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta):
    '''This function is valid for bonds that don't pay coupons, whereas the previous 
    two functions assume the bond pays coupons.
    Note: the field of `next_coupon_payment_date` corresponds to our variable of 
    `next_coupon_date` (removing the word `payment`) for more concise and readable 
    code, and similarly with `previous_coupon_date`.'''
    if frequency == 0:
        next_coupon_date = trade.maturity_date
        prev_coupon_date = trade.accrual_date
    else:
        if pd.isnull(trade.next_coupon_payment_date):
            next_coupon_date = get_next_coupon_date(trade.first_coupon_date, trade.settlement_date, time_delta)
        else:
            next_coupon_date = pd.to_datetime(trade.next_coupon_payment_date)

        if pd.isnull(trade.previous_coupon_payment_date):
            prev_coupon_date = get_previous_coupon_date(trade.first_coupon_date, trade.settlement_date, trade.accrual_date, time_delta, next_coupon_date)
        else:
            prev_coupon_date = pd.to_datetime(trade.previous_coupon_payment_date)
            strange = np.abs( diff_in_days_two_dates(prev_coupon_date, next_coupon_date - time_delta) ) > 1    # previous coupon payment date specified by the reference data seems to be incorrect
            strange = strange & (prev_coupon_date != trade.last_period_accrues_from_date)
            if strange: prev_coupon_date = next_coupon_date - time_delta

    return prev_coupon_date, next_coupon_date


def get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, end_date, time_delta):
    '''This function returns the number of interest payments and the final coupon 
    date based on the next coupon date, the end date, and the gap between coupon 
    payments. This function returns both together because one is always a 
    byproduct of computing the other.
    Note that the special case of an odd final coupon is handled below in 
    `price_of_bond_with_multiple_periodic_interest_payments`.'''
    if compare_dates(next_coupon_date, end_date) > 0:
        return 0, next_coupon_date    # return 1, end_date (would be valid in isolation)
    
    num_of_interest_payments = 1
    final_coupon_date = next_coupon_date
    while compare_dates(final_coupon_date + time_delta, end_date) <= 0:
        num_of_interest_payments += 1
        final_coupon_date += time_delta
    return num_of_interest_payments, final_coupon_date


def price_of_bond_with_interest_at_maturity(cusip,    # can be used for debugging purposes
                                            settlement_date, 
                                            accrual_date, 
                                            end_date, 
                                            yield_rate, 
                                            coupon, 
                                            RV):
    '''This function is called when the interest is only paid at maturity (which is represented 
    in the transformed dataframe as interest payment frequency equaling 0). There are two 
    cases when interest is paid at maturity. The first case is for short term bonds where 
    there is a single coupon payment at maturity, and this logic will reduce to the logic 
    in MSRB Rule Book G-33, rule (b)(i)(A). The second case is when when there is a compounding 
    accreted value (i.e., capital appreciation bonds) which accrues semiannually. Then, to get 
    the price of this bond, we need to account for the accrued interest. This can be thought 
    of as a bond that pays a coupon semiannually through the duration of the bond, but all the 
    coupon payments are made as a single payment at the time the bond is called / maturity. 
    For more info and an example, see the link: 
    https://www.investopedia.com/terms/c/cav.asp#:~:text=Compound%20accreted%20value%20(CAV)%20is,useful%20metric%20for%20bond%20investors.'''
    NOMINAL_FREQUENCY = 2    # semiannual interest payment frequency
    accrual_date_to_settlement_date = diff_in_days_two_dates(settlement_date, accrual_date)
    settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
    accrued = coupon * accrual_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
    num_of_periods_from_settlement_date_to_end_date = settlement_date_to_end_date / (NUM_OF_DAYS_IN_YEAR / NOMINAL_FREQUENCY)
    denom = (1 + yield_rate / NOMINAL_FREQUENCY) ** num_of_periods_from_settlement_date_to_end_date
    accrual_date_to_end_date = diff_in_days_two_dates(end_date, accrual_date)
    base = (RV + coupon * accrual_date_to_end_date / NUM_OF_DAYS_IN_YEAR) / denom
    return base - accrued


def price_of_bond_with_multiple_periodic_interest_payments(cusip,    # can be used for debugging purposes
                                                           settlement_date, 
                                                           accrual_date,
                                                           first_coupon_date, 
                                                           prev_coupon_date, 
                                                           next_coupon_date,    
                                                           final_coupon_date, 
                                                           end_date, 
                                                           frequency,
                                                           num_of_interest_payments, 
                                                           yield_rate,
                                                           coupon, 
                                                           RV, 
                                                           time_delta, 
                                                           last_period_accrues_from_date):
    '''This function computes the price of a bond with multiple periodic interest 
    payments using MSRB Rule Book G-33, rule (b)(i)(B)(2). Comments with capital 
    letter symbols represent those same symbols seen in formula in MSRB rule book.'''
    num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency
    discount_rate = 1 + yield_rate / frequency    # 1 + Y / M
    final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
    prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)    # A
    interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
    
    RV_and_interest_due_at_end_date = RV + interest_due_at_end_date
    settlement_date_to_next_coupon_date = diff_in_days_two_dates(next_coupon_date, settlement_date)    # E - A
    settlement_date_to_next_coupon_date_frac = settlement_date_to_next_coupon_date / num_of_days_in_period    # (E - A) / E
    final_coupon_date_to_end_date_frac = final_coupon_date_to_end_date / num_of_days_in_period
    num_of_periods_from_settlement_date_to_end_date = num_of_interest_payments - 1 + settlement_date_to_next_coupon_date_frac + final_coupon_date_to_end_date_frac
    
    RV_and_interest_due_at_end_date_discounted = RV_and_interest_due_at_end_date / (discount_rate ** num_of_periods_from_settlement_date_to_end_date)
    
    # The following logic statements are necessary to address odd first and final coupons
    if dates_are_equal(next_coupon_date, first_coupon_date):
        num_of_days_in_current_interest_payment_period = diff_in_days_two_dates(first_coupon_date, accrual_date)
    elif not pd.isna(last_period_accrues_from_date) and compare_dates(settlement_date, last_period_accrues_from_date + time_delta) > 0:    # this logic has not been tested
        num_of_days_in_current_interest_payment_period = 0
    else:
        num_of_days_in_current_interest_payment_period = num_of_days_in_period

    coupon_payments_discounted_total = (coupon * num_of_days_in_current_interest_payment_period / NUM_OF_DAYS_IN_YEAR) / \
                                       (discount_rate ** settlement_date_to_next_coupon_date_frac)
    coupon_payment = coupon / frequency
    for k in range(1, num_of_interest_payments):
        coupon_payment_discounted = coupon_payment / (discount_rate ** (settlement_date_to_next_coupon_date_frac + k))
        coupon_payments_discounted_total += coupon_payment_discounted
        
    accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR    # R * A / B
    return RV_and_interest_due_at_end_date_discounted + coupon_payments_discounted_total - accrued