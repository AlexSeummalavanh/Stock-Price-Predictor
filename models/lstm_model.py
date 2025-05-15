import tensorflow as tf

def build_model(input_shape):
    model = tf.keras.Sequential(name="Stock_Price_Predictor")
    model.add(tf.keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
