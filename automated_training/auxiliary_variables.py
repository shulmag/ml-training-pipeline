import os
import sys
from pytz import timezone
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR, PROJECT_ID, CATEGORICAL_FEATURES, CATEGORICAL_FEATURES_DOLLAR_PRICE, NON_CAT_FEATURES, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY, BINARY_DOLLAR_PRICE, PREDICTORS, PREDICTORS_DOLLAR_PRICE    # the unused imports here are used in `auxiliary_functions.py` and we import them here so that if we make modifications to them, then they will be preserved before the training procedure is called in `auxiliary_functions.py`


EASTERN = timezone('US/Eastern')

BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())    # used to skip over holidays when adding or subtracting business days

SAVE_MODEL_AND_DATA = True    # boolean indicating whether the trained model will be saved to google cloud storage; set to `False` if testing
USE_PICKLED_DATA = False    # boolean indicating whether the data used to train the model will be loaded from a local pickle file; set to `True` if testing

SENDER_EMAIL = 'notifications@ficc.ai'
EMAIL_RECIPIENTS_FOR_LOGS = ['ficc-eng@ficc.ai']    # recipients for training logs, which should be a more technical subset of `EMAIL_RECIPIENTS`
EMAIL_RECIPIENTS = EMAIL_RECIPIENTS_FOR_LOGS    # + ['myles@ficc.ai']    # recieve an email following a successful run of the training script; set to only your email if testing

BUCKET_NAME = 'automated_training'
TRAINING_LOGS_DIRECTORY = 'training_logs'
MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK = 10

YEAR_MONTH_DAY = '%Y-%m-%d'
HOUR_MIN_SEC = '%H:%M:%S'

MAX_NUM_DAYS_IN_THE_PAST_TO_KEEP_DATA = 270    # 270 = 9 * 30, so we are keeping approximately 9 months of data in the file (1 more month than we use as part of training so that we can create the trade history derived features); previously we were using 390 = 13 * 30 (approximately 13 months of data) in order to go beyond one year and allow for future experiments with annual patterns without having to re-create the entire dataset, but this was too expensive memory wise and was forcing us to use extra compute when training (2 GPUs instead of 1 GPU, etc.)
EARLIEST_TRADE_DATETIME = (datetime.now().date() - timedelta(days=MAX_NUM_DAYS_IN_THE_PAST_TO_KEEP_DATA)).strftime('%Y-%m-%d') + 'T00:00:00'    # must be a string for further downstream functions

HOME_DIRECTORY = os.path.expanduser('~')    # use of relative path omits the need to hardcode home directory; `os.path.expanduser('~')` parses `~` because pickle cannot read `~` as is
WORKING_DIRECTORY = f'{HOME_DIRECTORY}/ficc_python'

AUXILIARY_VIEWS_DATASET_NAME = 'auxiliary_views_v2'
HISTORICAL_PREDICTION_TABLE = {'yield_spread': f'{PROJECT_ID}.historic_predictions.historical_predictions', 
                               'yield_spread_with_similar_trades': f'{PROJECT_ID}.gs_hw.gil_test'}

MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME = {'yield_spread': 'processed_data_yield_spread.pkl', 
                                            'dollar_price': 'processed_data_dollar_price_v2.pkl', 
                                            'yield_spread_with_similar_trades': 'processed_data_yield_spread_with_similar_trades_v2.pkl'}

NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL = 5
NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL = 2

CATEGORICAL_FEATURES_VALUES = {'purpose_class' : list(range(53 + 1)),    # possible values for `purpose_class` are 0 through 53
                               'rating' : ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-',
                                           'BBB', 'BBB+', 'BBB-', 'CC', 'CCC', 'CCC+', 'CCC-' , 'D', 'NR', 'MR'],
                               'trade_type' : ['D', 'S', 'P'],
                               'incorporated_state_code' : ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
                                                            'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                                            'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                                            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'US', 'UT', 'VA', 'VI',
                                                            'VT', 'WA', 'WI', 'WV', 'WY'] }

TTYPE_DICT = {(0, 0): 'D', (0, 1): 'S', (1, 0): 'P'}

_VARIANTS = ['max_qty', 'min_ago', 'D_min_ago', 'P_min_ago', 'S_min_ago']
YS_VARIANTS = ['max_ys', 'min_ys'] + _VARIANTS
DP_VARIANTS = ['max_dp', 'min_dp'] + _VARIANTS
_FEATS = ['_ttypes', '_ago', '_qdiff']
YS_FEATS = ['_ys'] + _FEATS
DP_FEATS = ['_dp'] + _FEATS

LONG_TIME_AGO_IN_NUM_SECONDS = 9    # default `num_seconds_ago` value to signify that the trade was a long time ago (9 is a large value since the `num_seconds_ago` is log10 transformed)
MIN_TRADES_NEEDED_TO_BE_CONSIDERED_BUSINESS_DAY = 10000    # used to determine the minimum number of trades needed to be considered a "day of trades"; setting this value to 1 would check if any trades occurred that day

QUERY_FEATURES = ['rtrs_control_number',
                  'cusip',
                  'yield',
                  'is_callable',
                  'refund_date',
                  'accrual_date',
                  'dated_date',
                  'next_sink_date',
                  'coupon',
                  'delivery_date',
                  'trade_date',
                  'trade_datetime',
                  'par_call_date',
                  'interest_payment_frequency',
                  'is_called',
                  'is_non_transaction_based_compensation',
                  'is_general_obligation',
                  'callable_at_cav',
                  'extraordinary_make_whole_call',
                  'make_whole_call',
                  'has_unexpired_lines_of_credit',
                  'escrow_exists',
                  'incorporated_state_code',
                  'trade_type',
                  'par_traded',
                  'maturity_date',
                  'settlement_date',
                  'next_call_date',
                  'issue_amount',
                  'maturity_amount',
                  'issue_price',
                  'orig_principal_amount',
                  'max_amount_outstanding',
                  'recent',
                  'dollar_price',
                  'calc_date',
                  'purpose_sub_class',
                  'called_redemption_type',
                  'calc_day_cat',
                  'previous_coupon_payment_date',
                  'instrument_primary_name',
                  'purpose_class',
                  'call_timing',
                  'call_timing_in_part',
                  'sink_frequency',
                  'sink_amount_type',
                  'issue_text',
                  'state_tax_status',
                  'series_name',
                  'transaction_type',
                  'next_call_price',
                  'par_call_price',
                  'when_issued',
                  'min_amount_outstanding',
                  'original_yield',
                  'par_price',
                  'default_indicator',
                  'sp_long',
                #   'moodys_long',
                  'coupon_type',
                  'federal_tax_status',
                  'use_of_proceeds',
                  'muni_security_type',
                  'muni_issue_type',
                  'capital_type',
                  'other_enhancement_type',
                  'next_coupon_payment_date',
                  'first_coupon_date',
                  'last_period_accrues_from_date']
ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL = ['refund_price', 'publish_datetime', 'maturity_description_code']    # these features were used for testing, but are not needed, nonetheless, we keep them since the previous data files have these fields and `pd.concat(...)` will fail if the column set is different
ADDITIONAL_QUERY_FEATURES_FOR_YIELD_SPREAD_WITH_SIMILAR_TRADES_MODEL = ['recent_5_year_mat']

QUERY_CONDITIONS = ['par_traded >= 10000', 
                    'coupon_type in (8, 4, 10, 17)', 
                    'capital_type <> 10', 
                    'default_exists <> TRUE', 
                    # 'most_recent_default_event IS NULL', 
                    'default_indicator IS FALSE', 
                    'msrb_valid_to_date > current_date',    # condition to remove cancelled trades
                    'settlement_date IS NOT NULL']
ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL = ['yield IS NOT NULL', 'yield > 0']


OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD = {'use_treasury_spread': True, 
                                                    'only_dollar_price_history': False}
OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE = {'use_treasury_spread': False, 
                                                    'only_dollar_price_history': True}



if 'ficc_treasury_spread' not in PREDICTORS: PREDICTORS.append('ficc_treasury_spread')
if 'target_attention_features' not in PREDICTORS: PREDICTORS.append('target_attention_features')
if 'ficc_treasury_spread' not in NON_CAT_FEATURES: NON_CAT_FEATURES.append('ficc_treasury_spread')


# model training
NUM_EPOCHS = 100
BATCH_SIZE = 1000
DROPOUT = 0.01


MODEL_NAME_TO_ARCHIVED_MODEL_FOLDER = {'yield_spread': 'yield_spread_model', 
                                       'dollar_price': 'dollar_price_model', 
                                       'yield_spread_with_similar_trades': 'similar_trades_model'}


ROW_NAME_DETERMINING_MODEL_SWITCH = 'Investment Grade'


USE_END_OF_DAY_YIELD_CURVE_COEFFICIENTS = False    # boolean indicating whether to use the end-of-day yield curve coefficients; set to `False` if using the real-time (minute) yield curve coefficients


# setting variables for when `TESTING` is `True`
TESTING = False
if TESTING:
    SAVE_MODEL_AND_DATA = False
    USE_PICKLED_DATA = True
    NUM_EPOCHS = 2
    EARLIEST_TRADE_DATETIME = (datetime.now(EASTERN) - (BUSINESS_DAY * 2)).strftime(YEAR_MONTH_DAY) + 'T00:00:00'    # 2 business days before the current datetime (start of the day) to have enough days for training and testing; same logic as `auxiliary_functions::decrement_business_days(...)` but cannot import from there due to circular import issue
    print(f'In TESTING mode; SAVE_MODEL_AND_DATA=False and NUM_EPOCHS={NUM_EPOCHS} and EARLIEST_TRADE_DATETIME={EARLIEST_TRADE_DATETIME}')
    print('Check `get_creds(...)` to make sure the credentials filepath is correct')
    print('Check `WORKING_DIRECTORY` to make sure the path is correct')
    EMAIL_RECIPIENTS = EMAIL_RECIPIENTS_FOR_LOGS = ['ficc-eng@ficc.ai']
    print(f'Only sending emails to {EMAIL_RECIPIENTS}')
else:
    print(f'In PRODUCTION mode (to change to TESTING mode, set `TESTING` to `True`); all files and models will be saved and NUM_EPOCHS={NUM_EPOCHS}')
