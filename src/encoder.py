

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow # type: ignore

def create_transformer_encoder(input_shape, num_layers=4, d_model=64, num_heads=6, dff=128, dropout_rate=0.1): 
    inputs = keras.Input(shape=(input_shape)) 

    x = keras.layers.Reshape((input_shape, 1))(inputs)
    x = keras.layers.Dense(d_model)(x)

    x += keras.layers.Embedding(input_dim=input_shape, output_dim=d_model)(tensorflow.range(start=0, limit=input_shape, delta=1))  #positional encoding

    for _ in range(num_layers): 
        attn_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim = d_model)(x, x)
        attn_output = keras.layers.Dropout(dropout_rate)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x+attn_output)

        ffn = keras.Sequential([ # type: ignore
            keras.layers.Dense(dff, activation="relu"),
            keras.layers.Dense(d_model)
        ])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x+ffn(x))

    x = keras.layers.GlobalAveragePooling1D()(x) # it is used because we get the output of encoder to a classifier # type: ignore
    return keras.Model(inputs=inputs, outputs=x)

