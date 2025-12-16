'''
 # given the price.
 '''
import pandas as pd
import scipy.optimize as optimize

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import compare_dates
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.frequency import get_time_delta_from_interest_frequency, get_frequency
from ficc.utils.truncation import trunc_and_round_yield
from ficc.pricing.auxiliary_functions import get_num_of_interest_payments_and_final_coupon_date, \
                                             price_of_bond_with_multiple_periodic_interest_payments, \
                                             get_prev_coupon_date_and_next_coupon_date, \
                                             price_of_bond_with_interest_at_maturity
from ficc.pricing.called_trade import end_date_for_called_bond, refund_price_for_called_bond


def get_yield(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date,
              frequency, 
              price, 
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    '''This function is a helper function for `compute_yield`. This function calculates the yield of a trade given the price and other trade features, using
    the MSRB Rule Book G-33 linked at: https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.

    When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
    B: NUM_OF_DAYS_IN_YEAR
    E: number of days in interest payment period
    M: frequency
    N: num_of_interest_payments
    R: coupon'''
    settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)    # hold period days

    if frequency == 0:
        # See description for `price_of_bond_with_interest_at_maturity`
        yield_func = lambda yield_rate: -price + price_of_bond_with_interest_at_maturity(cusip, 
                                                                                         settlement_date, 
                                                                                         accrual_date, 
                                                                                         end_date, 
                                                                                         yield_rate, 
                                                                                         coupon, 
                                                                                         RV)
        yield_estimate = optimize.newton(yield_func, x0=0, maxiter=int(1e3))
    else:
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)    # accrued days from beginning of the interest payment period, used to be labelled `A`
        prev_coupon_date_to_end_date = diff_in_days_two_dates(end_date, prev_coupon_date)    # accrual days for final paid coupon
        num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency    # number of days in interest payment period 
        assert num_of_days_in_period == round(num_of_days_in_period)

        if compare_dates(end_date, next_coupon_date) <= 0:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(1)
            # Recall: number of interest payments per year * number of days in interest payment period = number of days in a year, i.e., E * M = NUM_OF_DAYS_IN_YEAR
            yield_estimate = (((RV + (coupon * prev_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR)) / \
                            (price + (coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR))) - 1) * \
                            (NUM_OF_DAYS_IN_YEAR / settlement_date_to_end_date)
        else:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(2)
            yield_func = lambda yield_rate: -price + price_of_bond_with_multiple_periodic_interest_payments(cusip, 
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
                                                                                                   last_period_accrues_from_date)
            yield_estimate = optimize.newton(yield_func, x0=0, maxiter=int(1e3))
    return trunc_and_round_yield(yield_estimate * 100)


def compute_yield(trade, price=None):
    '''This function computes the yield of a trade.'''
    if price == None:
        price = trade.dollar_price
    elif type(price) == str:
        raise ValueError('Price argument cannot be a string. It must be a numerical value.')

    frequency = get_frequency(trade.interest_payment_frequency)
    time_delta = get_time_delta_from_interest_frequency(frequency)
    my_prev_coupon_date, my_next_coupon_date = get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta)

    get_yield_caller = lambda end_date, redemption_value: get_yield(trade.cusip, 
                                                                    my_prev_coupon_date, 
                                                                    trade.first_coupon_date, 
                                                                    my_next_coupon_date,
                                                                    end_date, 
                                                                    trade.settlement_date, 
                                                                    trade.accrual_date, 
                                                                    frequency,
                                                                    price, 
                                                                    trade.coupon, 
                                                                    redemption_value, 
                                                                    time_delta, 
                                                                    trade.last_period_accrues_from_date)

    escrowed_to_maturity_but_callability_not_defeased = (trade.called_redemption_type in (1.0, 5.0)) and (trade.call_defeased != '1')    # explanation for this condition in section 'Bug Fix 4' of https://github.com/Ficc-ai/ficc/blob/ficc_ml/ml_models/sequence_predictors/yield_to_price_20230730.ipynb
    redemption_value_at_maturity = 100
    if (not trade.is_called) and (not trade.is_callable) and (not escrowed_to_maturity_but_callability_not_defeased):
        yield_to_maturity = get_yield_caller(trade.maturity_date, redemption_value_at_maturity)
        return yield_to_maturity, trade.maturity_date
    elif trade.is_called:
        end_date = end_date_for_called_bond(trade)

        if compare_dates(end_date, trade.settlement_date) < 0:
            print(f'Bond (CUSIP: {trade.cusip}) has an end date ({end_date}) which is before the settlement date ({trade.settlement_date}).')    # printing instead of raising an error to not disrupt processing large quantities of trades
            # raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).')

        redemption_value_at_refund = refund_price_for_called_bond(trade)
        yield_to_call = get_yield_caller(end_date, redemption_value_at_refund)
        return yield_to_call, end_date
    else:
        yield_to_par_call, yield_to_next_call, yield_to_maturity = float('inf'), float('inf'), float('inf')
        
        if not pd.isnull(trade.par_call_date) and trade.settlement_date < trade.par_call_date:
            yield_to_par_call = get_yield_caller(trade.par_call_date, trade.par_call_price)
        if not pd.isnull(trade.next_call_date) and trade.settlement_date < trade.next_call_date:
            yield_to_next_call = get_yield_caller(trade.next_call_date, trade.next_call_price)
        yield_to_maturity = get_yield_caller(trade.maturity_date, redemption_value_at_maturity)

        yield_rates_and_dates = [(yield_to_next_call, trade.next_call_date), 
                                 (yield_to_par_call, trade.par_call_date), 
                                 (yield_to_maturity, trade.maturity_date)]
        yield_to_worst, calc_date = min(yield_rates_and_dates, key=lambda pair: pair[0])    # the `min` function is stable and will choose the pair which appears first in the case of ties for the lowest yield
        return yield_to_worst, calc_date