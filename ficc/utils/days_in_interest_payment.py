'''
 #  the number of days in the interest payment period
 '''
from ficc.utils.auxiliary_variables import COUPON_FREQUENCY_TYPE, LARGE_NUMBER

def days_in_interest_payment(trade):
    frequency = COUPON_FREQUENCY_TYPE[trade['interest_payment_frequency']]
    if frequency == 0:
        # Changing it to large number from frequency = LARGE_NUMBER
        return LARGE_NUMBER
        
    return 360 / frequency