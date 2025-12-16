'''
 # The driver method for the package is the proces data function. 
 # The method takes the following arguments. 
 	#   1. A query that will be used to fetch data from BigQuery. 
	#   2. BigQuery client. 
 	#   3. The sequence length of the trade history can take 32 as its maximum value. 
	#   4. The number of features that the trade history contains. 
	#   5. Link to save the raw data grabbed from BigQuery. 
  #   6. The yield curve to use acceptable options are S&P, FICC, FICC_NEW, MMD and MSRB_YTW(to train estimating the yield). 
  #   7. remove_short_maturity flag to remove trades that mature within 400 days from trade date
  #   8. trade_history_delay flag to remove trades from history which occur within the specified minutes of the target trade
  #   9. min_trades_in_history the minimum number of trades allowed in the history
	#   10. A list containing the features that will be used for training. This is an optional parameter
 '''

import os
import time
from google.cloud import bigquery
from ficc.data.process_data import process_data

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/shayaan/ahmad_creds.json"

SEQUENCE_LENGTH = 8
NUM_FEATURES = 6

DATA_QUERY = '''SELECT
rtrs_control_number, 
cusip, 
yield, 
is_callable, 
refund_date,
refund_price,
accrual_date,
dated_date, 
next_sink_date,
coupon, 
delivery_date, 
trade_date, 
trade_datetime,
par_call_date, 
interest_payment_frequency,
is_called,
is_non_transaction_based_compensation,
is_general_obligation, 
callable_at_cav, 
extraordinary_make_whole_call,
make_whole_call, 
has_unexpired_lines_of_credit,
escrow_exists, 
incorporated_state_code,
trade_type, 
par_traded, 
maturity_date, 
settlement_date, 
next_call_date, 
issue_amount, 
maturity_amount, 
issue_price, 
orig_principal_amount,
publish_datetime,
max_amount_outstanding, 
recent,
recent_similar,
dollar_price,
calc_date,
purpose_sub_class,
called_redemption_type,
calc_day_cat, 
previous_coupon_payment_date,
instrument_primary_name, 
purpose_class,
call_timing,
call_timing_in_part,
sink_frequency,
sink_amount_type,
issue_text,
state_tax_status, 
series_name,
transaction_type,
next_call_price, 
par_call_price, 
when_issued,
min_amount_outstanding,
original_yield, 
par_price,
default_indicator,
sp_stand_alone,
sp_long, 
moodys_long, 
coupon_type,  
federal_tax_status,
use_of_proceeds, 
muni_security_type,
muni_issue_type,
capital_type, 
other_enhancement_type,  
next_coupon_payment_date,
first_coupon_date, 
last_period_accrues_from_date,
maturity_description_code 
FROM
`eng-reactor-287421.jesse_tests.new_similar_recent`
WHERE
  yield IS NOT NULL
  AND yield > 0
  AND par_traded >= 10000
  AND trade_date >= '2023-01-01'
  AND coupon_type in (8, 4, 10, 17)
  AND capital_type <> 10
  AND default_exists <> TRUE
  AND most_recent_default_event IS NULL
  AND default_indicator IS FALSE
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
  AND settlement_date is not null
  ORDER BY trade_datetime desc
  limit 100
'''


bq_client = bigquery.Client()


if __name__ == "__main__":
    start_time  = time.time()
    trade_data = process_data(DATA_QUERY, 
                              bq_client,
                              SEQUENCE_LENGTH,
                              NUM_FEATURES,
                              'data.pkl',
                              'FICC_NEW',
                              remove_short_maturity=False,
                              trade_history_delay = 0,
                              min_trades_in_history = 0,
                              treasury_spread = True,
                              add_flags=False,
                              add_related_trades_bool=False,
                              add_rtrs_in_history=False,
                              only_dollar_price_history = False)
    
    end_time = time.time()

    print(f"time elapsed in seconds = {end_time - start_time}")
    print(trade_data)
