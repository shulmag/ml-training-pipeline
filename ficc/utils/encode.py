'''
 '''
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


NOT_SEEN_IN_DATA_STRING = ' not_seen_in_data'    # used as a placeholder text for feature values that were not previously seen by the corresponding LabelEncoder; extra space in the beginning allows for it to be first in sorted order and therefore encoded as 0


def encode_and_get_encoders(df, binary_features, categorical_features, encoder_type='label', return_columns_instead_of_df=False):
    '''Encode all features in `features` with an encoder (specified by `encoder_type`) to 
    convert the features into integers. Creates another option for unseen values only for 
    `categorical_features`. If `label_or_one_hot` is 'label' '''
    assert encoder_type == 'label' or encoder_type == 'one_hot'

    if return_columns_instead_of_df: df_columns = df.copy()
    else: df_columns = pd.DataFrame(index=df.index)

    encoders = dict()
    binary_features_set = set(binary_features)
    for feature in binary_features + categorical_features:
        if not is_string_dtype(df[feature].dtype):
            df_columns[feature] = df[feature].astype('string')
        if feature in df:
            encoder = LabelEncoder() if encoder_type == 'label' or feature in binary_features_set else OneHotEncoder(sparse=False)    # sparse=False returns an array after transforming the values instead of a sparse matrix object
            series = df_columns[feature] if feature in df_columns.columns else df[feature]
            if encoder_type == 'one_hot': 
                series = series.to_numpy().reshape(-1, 1)    # need to reshape to avoid this error: `ValueError: Expected 2D array, got 1D array instead...Reshape your data either using array.reshape(-1, 1) if your data has a single feature`
            if feature in binary_features_set:
                series_to_fit = series
            else:    # feature must be a categorical feature
                assert NOT_SEEN_IN_DATA_STRING not in series, f'{NOT_SEEN_IN_DATA_STRING} is one of the feature values of the feature: {feature}'
                series_to_fit = pd.concat([series, pd.Series([NOT_SEEN_IN_DATA_STRING])]) if encoder_type == 'label' else np.append(series, [[NOT_SEEN_IN_DATA_STRING]], axis=0)    # add  a tag for a value that has not been seen in the training data to the end of the series
            encoder.fit(series_to_fit)
            transformed_values = encoder.transform(series)
            df_columns[feature] = transformed_values if encoder_type == 'label' else list(transformed_values)
            encoders[feature] = encoder
        else:
            warnings.warn(f'{feature} not found in the passed in dataframe')
    return df_columns, encoders


def encode_with_encoders(df, encoders, features_to_exclude=[]):
    '''Encode each feature in `df` by iterating through the dictionary `encoders`, where each 
    item has a key for the feature name and a value as the corresponding encoder. 
    `features_to_exclude` allows certain features to not be encoded.'''
    if type(features_to_exclude) == str: features_to_exclude = [features_to_exclude]    # if single feature passed in, then convert it to a one item list since `features_to_exclude` is handled as a list later on
    df_encoded = df.copy()
    for feature, encoder in encoders.items():
        if feature not in features_to_exclude:
            if type(encoder) == LabelEncoder: 
                encoder_type = 'label'
            elif type(encoder) == OneHotEncoder: 
                encoder_type = 'one_hot'
            else: 
                raise TypeError(f'Encoder type must be either LabelEncoder or OneHotEncoder, but was instead: {type(encoder)}')

            if not is_string_dtype(df_encoded[feature]): 
                df_encoded[feature] = df_encoded[feature].astype('string')
            unique_values = set(df_encoded[feature].unique())
            encoder_seen_values = encoder.classes_ if encoder_type == 'label' else encoder.categories_[0]
            
            for seen_value in encoder_seen_values:
                unique_values.discard(seen_value)
            unseen_values = list(unique_values)    # update the name to be more representative of the final list
            replaced_feature_values = df_encoded[feature].replace(unseen_values, NOT_SEEN_IN_DATA_STRING)
            
            if encoder_type == 'one_hot': 
                replaced_feature_values.to_numpy().reshape(-1, 1)    # need to reshape to avoid this error: `ValueError: Expected 2D array, got 1D array instead...Reshape your data either using array.reshape(-1, 1) if your data has a single feature`
            try:
                transformed_values = encoder.transform(replaced_feature_values)
                df_encoded[feature] = transformed_values if encoder_type == 'label' else list(transformed_values)
            except ValueError:    # output a more informative error message
                raise ValueError(f'{feature} has a value that was not found in its corresponding encoder.')
    return df_encoded