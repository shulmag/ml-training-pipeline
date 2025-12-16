'''
 # @ Create date: 2022-01-13
 # @ Modified date: 2024-12-04
 '''
import warnings
import pandas as pd

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import compare_dates
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.frequency import get_time_delta_from_interest_frequency, get_frequency
from ficc.utils.truncation import trunc_and_round_price
from ficc.pricing.auxiliary_functions import get_num_of_interest_payments_and_final_coupon_date, \
                                             price_of_bond_with_multiple_periodic_interest_payments, \
                                             get_prev_coupon_date_and_next_coupon_date, \
                                             price_of_bond_with_interest_at_maturity
from ficc.pricing.called_trade import end_date_for_called_bond, refund_price_for_called_bond


def get_price(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date, 
              frequency, 
              yield_rate,    # inputted as a percentage, e.g., 8%
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    '''This function is a helper function for `compute_price`. This function calculates the price of a trade, where `yield_rate` 
    is a specific yield and `end_date` is a fixed repayment date. All dates must be valid relative to the settlement 
    date, as opposed to the trade date. Note that 'yield' is a reserved word in Python and should not be used as the name 
    of a variable or column.

    Formulas are from https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.
    For all bonds, `base` is the present value of future cashflows to the buyer. 
    The clean price is this price minus the accumulated amount of simple interest that the buyer must pay to the seller, which is called `accrued`.
    Zero-coupon bonds are handled first. For these, the yield is assumed to be compounded semi-annually, i.e., once every six months.
    For bonds with non-zero coupon, the first and last interest payment periods may have a non-standard length,
    so they must be handled separately.

    When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
    A: prev_coupon_date_to_settlement_date
    B: NUM_OF_DAYS_IN_YEAR
    Y: yield_rate
    N: num_of_interest_payments
    E: num_of_days_in_period
    F: settlement_date_to_next_coupon_date
    P: price
    D: settlement_date_to_end_date
    H: prev_coupon_date_to_end_date
    R: coupon'''
    if yield_rate > 100: warnings.warn(f'yield_rate represents a percentage yield, but has a value of: {yield_rate}; perhaps it should have been divided by 100', RuntimeWarning)
    yield_rate = yield_rate / 100    # converted to a decimal value, e.g., an input of 8% would be converted to 0.08
    
    # Right now we do not disambiguate zero coupon from interest at maturity. More specfically, 
    # we should add logic that separates the cases of MSRB Rule Book G-33, rule (b) and rule (c)
    if frequency == 0:
        # See description for `price_of_bond_with_interest_at_maturity`
        price = price_of_bond_with_interest_at_maturity(cusip, 
                                                        settlement_date, 
                                                        accrual_date, 
                                                        end_date, 
                                                        yield_rate, 
                                                        coupon, 
                                                        RV)
    else:
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)
            
        num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency    # number of days in interest payment period 
        assert num_of_days_in_period == round(num_of_days_in_period), f'`frequency`: {frequency}; `num_of_days_in_period`: {num_of_days_in_period}; rounded version: {round(num_of_days_in_period)}'
         
        if compare_dates(end_date, next_coupon_date) <= 0:    # MSRB Rule Book G-33, rule (b)(i)(B)(1)
            if diff_in_days_two_dates(end_date, final_coupon_date) == 0:    # end date is the same (within one day) as the final coupon date, but the arithmetic later essentially includes one too many coupon payments; to fix this, one payment is subtracted from the redemption value (RV) and the final coupon date is shifted earlier
                final_coupon_date = prev_coupon_date
                RV = RV - coupon / frequency
            
            settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
            final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
            interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
            base = (RV + coupon / frequency + interest_due_at_end_date) / \
                   (1 + (yield_rate / frequency) * settlement_date_to_end_date / num_of_days_in_period)
            accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
            price = base - accrued
        else:    # MSRB Rule Book G-33, rule (b)(i)(B)(2)
            price = price_of_bond_with_multiple_periodic_interest_payments(cusip, 
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
    return trunc_and_round_price(price)


def compute_price(trade, yield_rate=None):
    '''This function computes the price of a trade. For bonds that have not been called, the price is the lowest of
    three present values: to the next call date (which may be above par), to the next par call date, and to maturity.'''
    if yield_rate == None:
        yield_rate = trade.ficc_ytw
    elif type(yield_rate) == str:
        raise ValueError('Yield rate argument cannot be a string. It must be a numerical value.')

    frequency = get_frequency(trade.interest_payment_frequency)
    time_delta = get_time_delta_from_interest_frequency(frequency)
    my_prev_coupon_date, my_next_coupon_date = get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta)

    get_price_caller = lambda end_date, redemption_value: get_price(trade.cusip, 
                                                                    my_prev_coupon_date, 
                                                                    trade.first_coupon_date, 
                                                                    my_next_coupon_date, 
                                                                    end_date, 
                                                                    trade.settlement_date, 
                                                                    trade.accrual_date, 
                                                                    frequency, 
                                                                    yield_rate, 
                                                                    trade.coupon, 
                                                                    redemption_value, 
                                                                    time_delta, 
                                                                    trade.last_period_accrues_from_date)

    called_redemption_type_is_1_or_5 = (not pd.isna(trade.called_redemption_type)) and (trade.called_redemption_type in (1.0, 5.0))
    call_defeased_is_not_1 = pd.isna(trade['call_defeased']) or (trade['call_defeased'] != '1')
    escrowed_to_maturity_but_callability_not_defeased = called_redemption_type_is_1_or_5 and call_defeased_is_not_1    # explanation for this condition in section 'Bug Fix 4' of https://github.com/Ficc-ai/ficc/blob/ficc_ml/ml_models/sequence_predictors/yield_to_price_20230730.ipynb
    
    redemption_value_at_maturity = 100
    if (not trade.is_called) and (not trade.is_callable) and (not escrowed_to_maturity_but_callability_not_defeased):
        yield_to_maturity = get_price_caller(trade.maturity_date, redemption_value_at_maturity)
        return yield_to_maturity, trade.maturity_date
    elif trade.is_called:
        end_date = end_date_for_called_bond(trade)

        if compare_dates(end_date, trade.settlement_date) < 0:
            print(f'Bond (CUSIP: {trade.cusip}) has an end date ({end_date}) which is before the settlement date ({trade.settlement_date}).')    # printing instead of raising an error to not disrupt processing large quantities of trades; remove RTRS control number since new pipeline does not have it
            # raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).')
        
        redemption_value_at_refund = refund_price_for_called_bond(trade)
        return get_price_caller(end_date, redemption_value_at_refund), end_date
    else:
        next_price, to_par_price, maturity_price = float('inf'), float('inf'), float('inf')

        if not pd.isnull(trade.par_call_date) and trade.settlement_date < trade.par_call_date:
            to_par_price = get_price_caller(trade.par_call_date, trade.par_call_price)
        if not pd.isnull(trade.next_call_date) and trade.settlement_date < trade.next_call_date:
            next_price = get_price_caller(trade.next_call_date, trade.next_call_price)
        maturity_price = get_price_caller(trade.maturity_date, redemption_value_at_maturity)

        prices_and_dates = [(next_price, trade.next_call_date), 
                            (to_par_price, trade.par_call_date), 
                            (maturity_price, trade.maturity_date)]
        calc_price, calc_date = min(prices_and_dates, key=lambda pair: pair[0])    # the `min` function is stable and will choose the pair which appears first in the case of ties for the lowest price
        return calc_price, calc_date
