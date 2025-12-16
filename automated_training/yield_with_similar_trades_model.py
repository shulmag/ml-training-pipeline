'''
'''
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from automated_training.auxiliary_variables import BATCH_SIZE, DROPOUT
from automated_training.set_random_seed import set_seed


set_seed()

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')    # currently this causes numerical instability; supposed to speed up deep learning training by using both 16-bit (float16) and 32-bit (float32) floating-point arithmetic instead of computing everything in float32; Weights & activations → Converted to float16, Loss & gradients → Kept in float32 for stability, Computation → Uses float16 for faster GPU execution, Automatic Casting → TensorFlow automatically handles mixed precision
# keras.backend.set_floatx('float32')    # currently does not make Apple Metal GPU more accurate; force float32 globally for higher accuracy


def model_definition(trade_history_normalizer, 
                     similar_trade_history_normalizer, 
                     noncat_binary_normalizer, 
                     num_trades_in_history, 
                     num_features_for_each_trade_in_history, 
                     categorical_features, 
                     non_cat_features, 
                     binary_features, 
                     fmax):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    
    similar_trade_history_input = layers.Input(name='similar_trade_history_input', 
                                               shape=(num_trades_in_history, num_features_for_each_trade_in_history), 
                                               dtype=tf.float32)
    
    # target_attention_input_for_similar_trade_history = layers.Input(name='target_attention_input_for_similar_trade_history', 
    #                                                                 shape=(1, 3), 
    #                                                                 dtype=tf.float32) 
    
    trade_history_input = layers.Input(name='trade_history_input', 
                                       shape=(num_trades_in_history, num_features_for_each_trade_in_history), 
                                       dtype=tf.float32)

    target_attention_input = layers.Input(name='target_attention_input', 
                                          shape=(1, 3), 
                                          dtype=tf.float32)


    inputs.append(similar_trade_history_input)
    # inputs.append(target_attention_input_for_similar_trade_history)
    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name='NON_CAT_AND_BINARY_FEATURES',
        shape=(len(non_cat_features + binary_features),)
    ))


    layer.append(noncat_binary_normalizer(inputs[-1]))

    ####################################################


    ############## SIMILAR TRADE HISTORY MODEL #################

    lstm_layer_similar_trades = layers.Bidirectional(layers.LSTM(50, 
                                                     activation='tanh',
                                                     input_shape=(num_trades_in_history, num_features_for_each_trade_in_history),
                                                     return_sequences=True,
                                                     name='LSTM_similar_trades'))

    lstm_layer_similar_trades_2 = layers.Bidirectional(layers.LSTM(100, 
                                                       activation='tanh',
                                                       input_shape=(num_trades_in_history, 50),
                                                       return_sequences=True,
                                                       name='LSTM_2_similar_trades'))



    similar_trade_history_features = lstm_layer_similar_trades(similar_trade_history_normalizer(inputs[0]))
    similar_trade_history_features = lstm_layer_similar_trades_2(similar_trade_history_features)
    similar_trade_history_features = layers.Flatten(name='similar_trade_history_features_flatten')(similar_trade_history_features)


    # similar_trade_history_attention_sequence = layers.Dense(200, activation='relu', name='similar_trade_history_attention_dense')(target_attention_input_for_similar_trade_history)
    # similar_trade_history_attention = layers.Dot(axes=[2, 2])([similar_trade_history_features, similar_trade_history_attention_sequence])
    # similar_trade_history_attention = layers.Activation('softmax')(similar_trade_history_attention)

    # similar_trade_history_context_vector = layers.Dot(axes=[1, 1])([similar_trade_history_features, similar_trade_history_attention])
    # similar_trade_history_context_vector = layers.Flatten(name='similar_trade_history_context_vector_flatten')(similar_trade_history_context_vector)
    
    # similar_trade_history_context_vector = layers.BatchNormalization()(similar_trade_history_context_vector)
    similar_trade_history_output = layers.Dense(100, activation='relu')(similar_trade_history_features)

    ####################################################


    ############## TRADE HISTORY MODEL #################

    lstm_layer = layers.Bidirectional(layers.LSTM(50, 
                                                  activation='tanh',
                                                  input_shape=(num_trades_in_history, num_features_for_each_trade_in_history),
                                                  return_sequences=True,
                                                  name='LSTM'))

    lstm_layer_2 = layers.Bidirectional(layers.LSTM(100, 
                                                    activation='tanh',
                                                    input_shape=(num_trades_in_history, 50),
                                                    return_sequences=True,
                                                    name='LSTM_2'))



    features = lstm_layer(trade_history_normalizer(inputs[1]))
    features = lstm_layer_2(features)


    attention_sequence = layers.Dense(200, activation='relu', name='attention_dense')(target_attention_input)
    attention = layers.Dot(axes=[2, 2])([features, attention_sequence])
    attention = layers.Activation('softmax')(attention)

    context_vector = layers.Dot(axes=[1, 1])([features, attention])
    context_vector = layers.Flatten(name='context_vector_flatten')(context_vector)

    context_vector = layers.BatchNormalization()(context_vector)
    trade_history_output = layers.Dense(100, activation='relu')(context_vector)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    for f in categorical_features:
        fin = layers.Input(shape=(1,), name=f)
        inputs.append(fin)
        embedded = layers.Flatten(name=f + '_flat')(layers.Embedding(input_dim=fmax[f] + 1, 
                                                                     output_dim=max(30, int(np.sqrt(fmax[f]))), 
                                                                     input_length=1, 
                                                                     name=f + '_embed')(fin))
        layer.append(embedded)

        
    reference_hidden = layers.Dense(400,
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer, axis=-1))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(DROPOUT)(reference_hidden)

    reference_hidden2 = layers.Dense(200, activation='relu', name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(DROPOUT)(reference_hidden2)

    reference_output = layers.Dense(100, activation='tanh', name='reference_hidden_3')(reference_hidden2)

    ####################################################

    feed_forward_input = layers.concatenate([reference_output, trade_history_output, similar_trade_history_output])

    hidden = layers.Dense(300, activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100, activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)
    return model


def yield_spread_with_similar_trades_model(x_train, 
                                           num_trades_in_history, 
                                           num_features_for_each_trade_in_history, 
                                           categorical_features, 
                                           non_cat_features, 
                                           binary_features,
                                           fmax):
    
    similar_trade_history_normalizer = Normalization(name='Similar_trade_history_normalizer')
    similar_trade_history_normalizer.adapt(x_train[0], batch_size=BATCH_SIZE)
    
    trade_history_normalizer = Normalization(name='Trade_history_normalizer')
    trade_history_normalizer.adapt(x_train[1], batch_size=BATCH_SIZE)

    noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
    noncat_binary_normalizer.adapt(x_train[3], batch_size=BATCH_SIZE)

    model = model_definition(trade_history_normalizer, 
                             similar_trade_history_normalizer, 
                             noncat_binary_normalizer, 
                             num_trades_in_history, 
                             num_features_for_each_trade_in_history, 
                             categorical_features, 
                             non_cat_features, 
                             binary_features,
                             fmax)
    return model
