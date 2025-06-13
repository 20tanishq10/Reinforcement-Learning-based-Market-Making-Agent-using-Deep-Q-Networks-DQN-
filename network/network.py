# network.py
import tensorflow as tf
from tensorflow.keras import layers, models

def compute_output_shape(input_shape):
    return tf.TensorShape([input_shape[0], 64])

def get_lob_model(hidden_dim=64, time_window=50):
    model = models.Sequential([
        layers.Input(shape=(time_window, 10*4*2)),  # 10 LOB levels, 4 types (bid dist, bid notional, ask dist, ask notional)
        layers.Conv1D(filters=hidden_dim, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(hidden_dim, activation='relu')
    ])
    return model

def get_fclob_model(hidden_dim=64, time_window=50):
    model = models.Sequential([
        layers.Input(shape=(time_window, 10*4*2)),
        layers.Flatten(),
        layers.Dense(hidden_dim, activation='relu'),
        layers.Dense(hidden_dim, activation='relu')
    ])
    return model

def get_pretrain_model(base_model, time_window):
    model = models.Sequential([
        base_model,
        layers.Dense(1, activation='linear')  # Regression output for price prediction
    ])
    return model

def get_model(lob_model, time_window, with_lob_state=True, with_market_state=True):
    inputs = []
    concat_list = []

    if with_lob_state:
        lob_input = layers.Input(shape=(time_window, 10*4*2), name="lob_input")
        lob_features = lob_model(lob_input)
        inputs.append(lob_input)
        concat_list.append(lob_features)

    if with_market_state:
        market_input = layers.Input(shape=(5,), name="market_input")  # e.g., midprice, spread, volume, etc.
        market_dense = layers.Dense(32, activation='relu')(market_input)
        inputs.append(market_input)
        concat_list.append(market_dense)

    if len(concat_list) > 1:
        x = layers.Concatenate()(concat_list)
    else:
        x = concat_list[0]

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model
