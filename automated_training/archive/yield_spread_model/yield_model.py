'''
 # @ Create date: 2023-02-01
 # @ Modified date: 2024-02-14
 '''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from auxiliary_variables import BATCH_SIZE, DROPOUT
from set_random_seed import set_seed
# from custom_attention import CustomAttention


set_seed()


def model_definition(trade_history_normalizer,
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
    trade_history_input = layers.Input(name='trade_history_input', 
                                       shape=(num_trades_in_history, num_features_for_each_trade_in_history), 
                                       dtype=tf.float32) 

    target_attention_input = layers.Input(name='target_attention_input', 
                                          shape=(1, 3), 
                                          dtype=tf.float32) 


    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name='NON_CAT_AND_BINARY_FEATURES',
        shape=(len(non_cat_features + binary_features),)
    ))


    layer.append(noncat_binary_normalizer(inputs[2]))
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



    features = lstm_layer(trade_history_normalizer(inputs[0]))
    features = lstm_layer_2(features)


    attention_sequence = layers.Dense(200, activation='relu', name='attention_dense')(target_attention_input)
    attention = layers.Dot(axes=[2, 2])([features, attention_sequence])
    attention = layers.Activation('softmax')(attention)

    context_vector = layers.Dot(axes=[1, 1])([features, attention])
    context_vector = layers.Flatten(name='context_vector_flatten')(context_vector)

    context_vector = layers.BatchNormalization()(context_vector)
    trade_history_output = layers.Dense(100,activation='relu')(context_vector)

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

    feed_forward_input = layers.concatenate([reference_output, trade_history_output])

    hidden = layers.Dense(300, activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100, activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)
    return model


def yield_spread_model(x_train, 
                       num_trades_in_history, 
                       num_features_for_each_trade_in_history, 
                       categorical_features, 
                       non_cat_features, 
                       binary_features,
                       fmax):
    
    trade_history_normalizer = Normalization(name='Trade_history_normalizer')
    trade_history_normalizer.adapt(x_train[0], batch_size=BATCH_SIZE)

    noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
    noncat_binary_normalizer.adapt(x_train[2], batch_size=BATCH_SIZE)

    model = model_definition(trade_history_normalizer,
                             noncat_binary_normalizer, 
                             num_trades_in_history, 
                             num_features_for_each_trade_in_history, 
                             categorical_features, 
                             non_cat_features, 
                             binary_features,
                             fmax)
    return model
