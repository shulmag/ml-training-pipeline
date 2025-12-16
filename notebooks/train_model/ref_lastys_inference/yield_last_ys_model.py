"""
A simplified yield spread model.
This model predicts yield spread using only reference data and last trade's yield spread.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from automated_training.auxiliary_variables import BATCH_SIZE, DROPOUT
from automated_training.set_random_seed import set_seed

set_seed()


def model_definition(last_ys_normalizer,
                     noncat_binary_normalizer,
                     categorical_features,
                     non_cat_features,
                     binary_features,
                     fmax):
    inputs = []
    layer = []

    # last yield spread input
    last_ys_input = layers.Input(name="last_yield_spread_input", shape=(1,), dtype=tf.float32)
    inputs.append(last_ys_input)
    layer.append(last_ys_normalizer(last_ys_input))

    # numerical + binary reference data
    ref_input = layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(non_cat_features + binary_features),),
    )
    inputs.append(ref_input)
    layer.append(noncat_binary_normalizer(ref_input))

    # categorical reference data
    for f in categorical_features:
        fin = layers.Input(shape=(1,), name=f)
        inputs.append(fin)
        embedded = layers.Flatten(name=f + "_flat")(layers.Embedding(
            input_dim=fmax[f] + 1,
            output_dim=max(30, int(np.sqrt(fmax[f]))),
            input_length=1,
            name=f + "_embed",
        )(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(400, activation="relu", name="reference_hidden_1")(
        layers.concatenate(layer, axis=-1)
    )
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(DROPOUT)(reference_hidden)

    reference_hidden2 = layers.Dense(200, activation="relu", name="reference_hidden_2")(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(DROPOUT)(reference_hidden2)

    reference_output = layers.Dense(100, activation="tanh", name="reference_hidden_3")(reference_hidden2)

    hidden = layers.Dense(300, activation="relu")(reference_output)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100, activation="tanh")(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)
    return model


def yield_last_ys_model(
    x_train,
    categorical_features,
    non_cat_features,
    binary_features,
    fmax,
):
    last_ys_normalizer = Normalization(name="last_yield_spread_normalizer")
    last_ys_normalizer.adapt(x_train[0], batch_size=BATCH_SIZE)

    noncat_binary_normalizer = Normalization(name="Numerical_binary_normalizer")
    noncat_binary_normalizer.adapt(x_train[1], batch_size=BATCH_SIZE)

    model = model_definition(
        last_ys_normalizer,
        noncat_binary_normalizer,
        categorical_features,
        non_cat_features,
        binary_features,
        fmax,
    )
    return model