'''
 '''
from collections import deque, defaultdict
import warnings
import multiprocess as mp    # using `multiprocess` instead of `multiprocessing` because function to be called in `map` is in the same file as the function which is calling it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

import numpy as np
import pandas as pd
from tqdm import tqdm

from ficc.utils.auxiliary_variables import VERY_LARGE_NUMBER, \
                                           NUM_OF_DAYS_IN_YEAR, \
                                           RELATED_TRADE_FEATURE_FUNCTIONS, \
                                           RELATED_TRADE_CATEGORICAL_FEATURES, \
                                           get_neighbor_feature, \
                                           get_appended_feature_name, \
                                           related_trade_features,\
                                           flatten

from ficc.utils.auxiliary_functions import compare_dates, double_quote_a_string
from ficc.utils.trade_mapping import TRADE_TYPE_MAPPING, TRADE_TYPE_CROSS_PRODUCT_MAPPING
from ficc.utils.encode import encode_and_get_encoders


RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED = 'rating_without_+-_b_nr_combined'
DAYS_TO_MATURITY_CATEGORICAL = 'days_to_maturity_categorical'
DAYS_TO_CALL_CATEGORICAL = 'days_to_call_categorical'
COUPON_CATEGORICAL = 'coupon_categorical'
PURPOSE_CLASS_TOP_VALUES = 'purpose_class_top_values'
quantized_features = [RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, 
                      DAYS_TO_MATURITY_CATEGORICAL, 
                      DAYS_TO_CALL_CATEGORICAL, 
                      COUPON_CATEGORICAL, 
                      PURPOSE_CLASS_TOP_VALUES]


related_trades_criterion_opt = ['incorporated_state_code', RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED, DAYS_TO_MATURITY_CATEGORICAL, DAYS_TO_CALL_CATEGORICAL, COUPON_CATEGORICAL, PURPOSE_CLASS_TOP_VALUES], []


def _recent_trade_data_subset(df, 
                              num_recent_trades, 
                              appended_features_names_and_functions, 
                              categories=[], 
                              filtering_conditions=[], 
                              header=None):
    sorted_df = df.sort_values(by='trade_datetime')
    header = (header,) if len(categories) == 1 else header    # puts `header` into a 1-dimensional tuple so it is a tuple type; function is used so that any header is of type tuple
    appended_features_names = list(appended_features_names_and_functions.keys())
    num_appended_features = len(appended_features_names)

    # need this if...else logic since there could be trades here that do not have the same header due to the similarity function, i.e., if no similarity function is involved, then `len(sorted_df)` equals `eval(conditions_as_string).sum()`; this boolean condition is stored in `all_rows_match_header`
    if categories:
        conditions_as_string = ' & '.join([f'(sorted_df[{double_quote_a_string(name)}] == {double_quote_a_string(value)})' for name, value in zip(categories, header)])    # e.g., if categories = [incorporated_state_code, rating] and header = ('CA', 'A+'), then conditions_as_string = (sorted_df['incorporated_state_code'] == 'CA') & (sorted_df['rating'] == 'A+'); the tuple wrapper on the header is for the edge case of having a single category
        len_augmented_data = eval(conditions_as_string).sum()
    else:
        len_augmented_data = len(sorted_df)
    all_rows_match_header = len_augmented_data == len(sorted_df)

    # initialization of augmented_data matrix
    augmented_data = np.zeros(shape=(len_augmented_data, 1 + num_recent_trades * num_appended_features), dtype=np.float32)    # first item is row index and rest are trade features of the past trades
    for appended_feature_idx, (_, init) in enumerate(appended_features_names_and_functions.values()):
        if init != 0: augmented_data[:, np.array(range(num_recent_trades)) * num_appended_features + 1 + appended_feature_idx] = init

    cusip_to_past_trade_info = dict()    # key: cusip, value: (sorted_df index, row corresponding to cusip); used to cache the work already done for finding the related trades for a cusip 
    all_recent_trades = deque([])    # using a deque for constant time `appendleft` which allows us to iterate from most recent to least recent when iterating through `recent_trades`
    idx_adjustment = 0    # this index adjustment is for when `sorted_df` has moved forward to another row that is not one of the rows that match `header` (but is a "similar" one)
    for idx, (row_idx, row) in enumerate(sorted_df.iterrows()):
        if all_rows_match_header or tuple(row[categories]) == header:
            row_past_trade_info = augmented_data[idx - idx_adjustment, 1:]
            augmented_data[idx - idx_adjustment, 0] = row_idx    # put the row index in the first column of `augmented_data`
            num_recent_trades_augmented = 0
            row_cusip = row['cusip']
            
            idx_already_seen, already_seen_past_trade_info = cusip_to_past_trade_info.get(row_cusip, (None, None))    # this index has already been processed; `(None, None)` is the default value if `row['cusip']` is not found
            for recent_trade_idx, neighbor in all_recent_trades:
                if recent_trade_idx == idx_already_seen:
                    num_recent_trades_left_to_be_augmented = num_recent_trades - num_recent_trades_augmented
                    row_past_trade_info[num_recent_trades_augmented * num_appended_features:] = already_seen_past_trade_info[:num_recent_trades_left_to_be_augmented * num_appended_features]
                    break    # we have already filled the `num_recent_trades` recent trades for the current trade (`row`)
                elif compare_dates(row['trade_datetime'], neighbor['trade_datetime']) > 0 and row_cusip != neighbor['cusip']:    # ensure that recent trades do not come from the same CUSIP, since that is handled by a separate past trade sequence; TODO: change first condition to only check for past trades greater than 60 seconds from the target trade (current implementation is 0 seconds), which will cause changes to memoization 
                    starting_position = num_recent_trades_augmented * num_appended_features
                    row_past_trade_info[starting_position : starting_position + num_appended_features] = np.array([appended_features_function(row, neighbor) for appended_features_function, _ in appended_features_names_and_functions.values()])
                    num_recent_trades_augmented += 1
                    if num_recent_trades_augmented == num_recent_trades: break    # going in here means that we have already filled the `num_recent_trades` recent trades for the current trade (`row`)

            augmented_data[idx - idx_adjustment, 1:] = row_past_trade_info
            cusip_to_past_trade_info[row_cusip] = (idx, row_past_trade_info)
        else:
            idx_adjustment += 1

        if all([condition(row) for condition in filtering_conditions]): all_recent_trades.appendleft((idx, row))    # only consider trades that meet the `filtering_conditions`
    
    return augmented_data


def append_recent_trade_data(df, 
                             num_recent_trades, 
                             appended_features_names_and_functions, 
                             feature_prefix='', 
                             categories=[], 
                             filtering_conditions=[], 
                             is_similar=None, 
                             return_df=False, 
                             hide_tqdm=True, 
                             multiprocessing=False, 
                             df_for_related_trades=None, 
                             return_new_columns_only=False):
    '''This function takes in a dataframe (`df`), a number of recent trades (`num_recent_trades`), a list 
    of categories (`categories`), and a similarity function (`is_similar`). The function
    `is_similar` is a similarity function which takes in two tuples of categories and 
    returns True iff the categories are considered similar by the function. The goal is 
    to augment each trade with previous trades that are similar to this one, where the 
    `is_similar` function determines whether two trades are similar. If `is_similar` is 
    equal to None, then this is equivalent to the similarity function enforcing that 
    all categories amongst two trades must be equal in order to be considered similar. 
    `filtering_conditions` is a list of filtering conditions which filter the recent trades 
    which may be considered to be appended. Note that each filtering condition is a function on 
    a row of the dataframe, e.g., if we wanted all related trades to have a quantity >= 10000, 
    our condition would be `lambda: row: row['quantity'] >= np.log(10000)'. `df_for_related_trades` 
    allows the augmented data to be encoded (or transformed in some other way) since the procedure will 
    populate the augmented data from `df_for_related_trades` if it is not None.'''
    assert 'trade_datetime' in df.columns, 'trade_datetime column is required'    # we use the datetime to determine the time of the trade
    if df_for_related_trades is None: 
        df_for_related_trades = df
    else:    # replace columns in df with columns from df_for_related_trades
        columns_in_df_and_df_for_related_trades = list(set(df.columns) & set(df_for_related_trades.columns))
        df_for_related_trades = pd.concat((df.drop(columns=columns_in_df_and_df_for_related_trades), df_for_related_trades), axis=1)

    if multiprocessing and not hide_tqdm: warnings.warn('Cannot use TQDM progress bar with multiprocessing, so no progress bar will be visible.')

    if categories:    # entering here means that there are certain categories that need to be similar between two trades to be considered amongst the recent trades
        _recent_trade_data_subset_caller = lambda header, df: _recent_trade_data_subset(df, num_recent_trades, appended_features_names_and_functions, categories, filtering_conditions, header)
        # `categories_as_multi_item_list` is created to avoid the following warning when passing in a one item list to the `.groupby(...)`: FutureWarning: In a future version of pandas, a length 1 tuple will be returned when iterating over a groupby with a grouper equal to a list of length 1. Don't supply a list with a single grouper to avoid this warning.
        categories_as_multi_item_list = categories
        if len(categories) == 1: categories_as_multi_item_list = categories[0]

        if is_similar is not None:    # entering here means that there is a similarity function
            subcategory_headers = []
            subcategory_dict = dict()
            for subcategory_header, subcategory_df in df_for_related_trades.groupby(categories_as_multi_item_list, observed=True):
                if type(subcategory_header) != tuple: subcategory_header = (subcategory_header,)    # this if statement converts a single item category value to a tuple to be consistent with the case when there are multiple categories
                subcategory_headers.append(subcategory_header)
                subcategory_dict[subcategory_header] = subcategory_df

            subcategory_header_to_related_subcategories_df_dict = dict()
            for subcategory_header in tqdm(subcategory_headers, disable=hide_tqdm):
                related_subcategories_df = [subcategory_dict[other_subcategory_header] for other_subcategory_header in subcategory_headers if is_similar(categories, subcategory_header, other_subcategory_header)]    # check if each subcategory header is similar to any of the other subcategory headers (trivally similar to itself)
                related_subcategories_df = pd.concat(related_subcategories_df)
                subcategory_header_to_related_subcategories_df_dict[subcategory_header] = related_subcategories_df

            if multiprocessing:
                with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                    augmented_data = pool_object.starmap(_recent_trade_data_subset_caller, subcategory_header_to_related_subcategories_df_dict.items())    # need to use starmap since `_recent_trade_data_subset_caller` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
            else:
                related_subcategories_df = subcategory_header_to_related_subcategories_df_dict[subcategory_header]
                augmented_data = [_recent_trade_data_subset_caller(subcategory_header, related_subcategories_df) for subcategory_header in tqdm(subcategory_headers, disable=hide_tqdm)]
        else:    # entering here means that there is no similarity function and all `categories` must be the same for two trades to be considered similar
            if multiprocessing:
                with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                    augmented_data = pool_object.starmap(_recent_trade_data_subset_caller, df_for_related_trades.groupby(categories_as_multi_item_list, observed=True))    # need to use starmap since `_recent_trade_data_subset_caller` has multiple arguments: https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
            else:
                augmented_data = [_recent_trade_data_subset_caller(subcategory_header, subcategory_df) for subcategory_header, subcategory_df in tqdm(df_for_related_trades.groupby(categories_as_multi_item_list, observed=True), disable=hide_tqdm)]
        augmented_data = np.concatenate(augmented_data, axis=0)
    else:
        augmented_data = _recent_trade_data_subset(df_for_related_trades, num_recent_trades, appended_features_names_and_functions, filtering_conditions=filtering_conditions)
    
    augmented_data = augmented_data[augmented_data[:, 0].argsort()]    # first column is the trade index, so sort by that
    all_appended_features_names = flatten([[get_appended_feature_name(trade_idx, feature_name, feature_prefix) for feature_name in appended_features_names_and_functions] for trade_idx in range(num_recent_trades)])    # populate all the columns names for the appended features; insertion order of the dictionary is preserved for Python v3.7+
    df_from_augmented_data = pd.DataFrame(augmented_data[:, 1:], columns=all_appended_features_names, index=df.index)    # set the index to be the same as the original dataframe in order to allow concatenation without NaN values (which occur when the index does not match)
    
    if return_df:
        if return_new_columns_only: return df_from_augmented_data
        else: return pd.concat((df, df_from_augmented_data), axis=1)


def convert_trade_type_encoding_to_actual(df, num_trades_in_history, new_column_name='trade_type', prefix='last_'):
    '''Replace every encoded trade_type value in a flattened `df` (without reference data) 
    with the decoded actual trade_type. Using subtraction as the aggregation method, since 
    this is one-to-one.'''
    decoded_trade_type_map = {2 * value1 - value2 : trade_type for trade_type, (value1, value2) in TRADE_TYPE_MAPPING.items()}
    assert len(decoded_trade_type_map) == len(TRADE_TYPE_MAPPING), 'Aggregation procedure is not one-to-one'

    df_columns_set = set(df.columns)
    old_columns, new_columns = [], []
    for trade_idx in range(num_trades_in_history):
        get_trade_type_column_name = lambda num: get_appended_feature_name(trade_idx, f'trade_type{num}', prefix)
        trade_type1_column_name, trade_type2_column_name = get_trade_type_column_name(1), get_trade_type_column_name(2)
        old_columns.extend([trade_type1_column_name, trade_type2_column_name])
        trade_type_column_name = get_appended_feature_name(trade_idx, new_column_name, prefix)
        assert trade_type_column_name not in df_columns_set, f'Need to use the `{trade_type_column_name}` column to store the results of the decoding'
        new_columns.append(trade_type_column_name)
        with warnings.catch_warnings():    # temporarily suppress warnings for a block of code: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
            warnings.simplefilter('ignore')    # suppress this warning for the line below: `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead`
            df.loc[:, trade_type_column_name] = ((2 * df[trade_type1_column_name] - df[trade_type2_column_name]).map(decoded_trade_type_map.get)).astype('category')    # change type to category so LightGBM can use this feature as a categorical feature
    return df, old_columns, new_columns


def add_related_trades(df, prefix, num_trades, categorical_features, default_values=None):
    # df = df.copy()
    if default_values is None: 
        default_values = defaultdict(int)

    RELATED_TRADE_CATEGORICAL_FEATURES_WITH_REFERENCE_FEATURES = list(set(RELATED_TRADE_CATEGORICAL_FEATURES + categorical_features))
    categorical_features_to_add_functions = {feature: get_neighbor_feature(feature) for feature in categorical_features}
    related_trade_feature_functions = {**RELATED_TRADE_FEATURE_FUNCTIONS, **categorical_features_to_add_functions}    # RELATED_TRADE_FEATURE_FUNCTIONS | categorical_features_to_add_functions    # changed code to work with python 3.7 (| notation for combining ditionaries is for 3.9+)

    # print(f'Each trade in the related trade history has the following features: {list(related_trade_feature_functions.keys())}')

    RELATED_TRADE_FEATURE_FUNCTIONS_AND_DEFAULT_VALUES = {key: (function, default_values[key]) for key, function in related_trade_feature_functions.items()}

    epsilon = 1 / VERY_LARGE_NUMBER

    df[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = df['rating'].transform(lambda rating: str.rstrip(rating, '+-'))    # remove + and - from right side of string
    # group BBB, BB, B, and NR together since each have a very small number of trades
    b_ratings = df[RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED].isin(['B', 'BB', 'BBB', 'NR'])
    df.loc[b_ratings, RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED] = 'B'
    print(f'Created {RATING_WITHOUT_PLUS_MINUS_B_NR_COMBINED} feature')

    num_of_days_bins_maturity = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, NUM_OF_DAYS_IN_YEAR * 10, VERY_LARGE_NUMBER]]    # 2 years, 5 years, 10 years; arbitrarily chosen
    df[DAYS_TO_MATURITY_CATEGORICAL] = pd.cut(df['days_to_maturity'], num_of_days_bins_maturity).astype('string')
    print(f'Created {DAYS_TO_MATURITY_CATEGORICAL} feature')

    num_of_days_bins_call = [np.log10(days) for days in [epsilon, NUM_OF_DAYS_IN_YEAR * 2, NUM_OF_DAYS_IN_YEAR * 5, VERY_LARGE_NUMBER]]    # 2 years, 5 years; arbitrarily chosen
    df[DAYS_TO_CALL_CATEGORICAL] = pd.cut(df['days_to_call'], num_of_days_bins_call).astype('string')
    print(f'Created {DAYS_TO_CALL_CATEGORICAL} feature')

    df[COUPON_CATEGORICAL] = pd.cut(df['coupon'], coupon_bins, right=False).astype('string')
    print(f'Created {COUPON_CATEGORICAL} feature')

    df[PURPOSE_CLASS_TOP_VALUES] = df['purpose_class']
    top6_purpose_class_values = df['purpose_class'].value_counts().head(6).index.tolist()    # select the top 6 coupon values based on frequency in the data, which are: 37 (school district), 51 (various purpose), 50 (utility), 46 (tax revenue), 9 (education), 48 (transportation) comprising about 80% of the data
    df.loc[~df['purpose_class'].isin(top6_purpose_class_values), PURPOSE_CLASS_TOP_VALUES] = -1    # arbitrary numerical value that is invalid as a purpose_class value
    print(f'Created {PURPOSE_CLASS_TOP_VALUES} feature')

    categories_to_match, filtering_conditions = related_trades_criterion_opt

    df_encoded_columns, encoders = encode_and_get_encoders(df, [], categorical_features)

    print(f'Putting related trades into data')
    related_trades_columns = append_recent_trade_data(df, 
                                                      num_trades, 
                                                      RELATED_TRADE_FEATURE_FUNCTIONS_AND_DEFAULT_VALUES, 
                                                      feature_prefix=prefix, 
                                                      categories=categories_to_match, 
                                                      filtering_conditions=filtering_conditions, 
                                                      return_df=True, 
                                                      multiprocessing=True, 
                                                      df_for_related_trades=df_encoded_columns, 
                                                      return_new_columns_only=True)
    df = pd.concat((df, related_trades_columns), axis=1).drop(columns=quantized_features)    # drop the quantized features from the final dataframe

    # convert same_day features to boolean
    same_day_features = related_trade_features(['same_day'])
    df.loc[:, same_day_features] = df.loc[:, same_day_features].astype(bool)

    # convert categorical features to original value
    for feature in categorical_features:
        related_trade_features_for_feature = related_trade_features([feature])
        encoder = encoders[feature]
        for related_trade_feature in related_trade_features_for_feature:
            df.loc[:, related_trade_feature] = encoder.inverse_transform(df.loc[:, related_trade_feature].astype(int))    # change type to int to work with `.inverse_transform(...)` function of `LabelEncoder`
        df.loc[:, related_trade_features_for_feature] = df.loc[:, related_trade_features_for_feature].astype('category')

    # convert trade_type1 and trade_type2 features to trade_type categorical feature
    df, old_trade_type_columns, new_trade_type_features = convert_trade_type_encoding_to_actual(df, num_trades, prefix=prefix)    # category dtype conversion already occurs in this function
    df = df.drop(columns=old_trade_type_columns)

    # convert trade_type_past_latest features to categorical
    trade_type_past_latest_features = related_trade_features(['trade_type_past_latest'])
    inverted_trade_type_cross_product_mapping = {value: key for key, value in TRADE_TYPE_CROSS_PRODUCT_MAPPING.items()}
    for feature in trade_type_past_latest_features:
        df.loc[:, feature] = df.loc[:, feature].map(inverted_trade_type_cross_product_mapping.get)
    df.loc[:, trade_type_past_latest_features] = df.loc[:, trade_type_past_latest_features].astype('category')

    return df 