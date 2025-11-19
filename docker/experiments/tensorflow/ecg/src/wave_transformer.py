import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class WaveLayer(tf.keras.Layer):
    def __init__(self, wave_size, **kwargs):
        super().__init__(**kwargs)
        self.wave_size = wave_size

    def build(self, input_shape):
        self.waves = self.add_weight(
            name='waves',
            shape=(self.wave_size, self.wave_size),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x, stride=1):
        Q, _ = tf.linalg.qr(self.waves)
        kernel = tf.reshape(Q, shape=(self.wave_size, 1, self.wave_size))
        return tf.nn.conv1d(x, filters=kernel, stride=stride, padding='VALID')

class WaveTransformer(tf.keras.Model):
    def __init__(self, window_size, wave_size, hidden_units, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.wave_size = wave_size
        self.stride = stride

        self.wave_layer = WaveLayer(wave_size)
        self.nn = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(hidden_units, activation='relu'),
                tf.keras.layers.Dense(wave_size)
            ],
            name='wave_space_nn'
        )
    
    def call(self, x):
        # transform into the wave space
        z = self.wave_layer(x)

        # average wave amplitudes
        z = tf.reduce_mean(z, axis=1)

        # reconstruct the wave from the excited wave amplitudes
        z = self.nn(z)

        # transform into the time space
        Q, _ = tf.linalg.qr(self.wave_layer.waves)
        z = tf.matmul(z, tf.transpose(Q))

        return tf.expand_dims(z, axis=-1)
    
    def wave_transform(self, x, pooling='avg'):
        x = tf.cast(x, dtype=tf.float32)

        Q, _ = tf.linalg.qr(self.waves)

        # transform into the wave space
        kernel = tf.reshape(Q, shape=(self.wave_size, 1, self.wave_size))
        z = tf.nn.conv1d(x, filters=kernel, stride=self.stride, padding='VALID')

        if (pooling == 'avg'):
            z = tf.reduce_mean(z, axis=1)

        return z
    
    def wave_transform_inverse(self, x):
        x = tf.cast(x, dtype=tf.float32)

        Q, _ = tf.linalg.qr(self.waves)

        return tf.matmul(x, tf.transpose(Q))

def plot_results(model, x, y_true):
    y_pred = model(x)
    x = np.squeeze(x)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    results = zip(x, y_pred, y_true)

    batch_size, window_size = x.shape
    _, wave_size = y_true.shape

    _, axes = plt.subplots(batch_size, 3, figsize=(9, batch_size))

    for (x_i, y_pred_i, y_true_i), (ax_i, ay_pred_i, ay_true_i) in zip(results, axes):
        ax_i.plot(np.arange(window_size), x_i)
        ay_pred_i.plot(np.arange(wave_size), y_pred_i)
        ay_true_i.plot(np.arange(wave_size), y_true_i)

    plt.tight_layout()
    plt.show()

def plot_wave_transform(model, x, y_true):
    y_pred = model(x)

    x_amp = model.wave_transform(x)
    y_pred_amp = model.wave_transform(y_pred)
    y_true_amp = model.wave_transform(y_true)

    amps = zip(x_amp, y_pred_amp, y_true_amp)

    batch_size, wave_size = x_amp.shape
    _, axes = plt.subplots(batch_size, 3, figsize=(9, batch_size))

    for (x_i, y_pred_i, y_true_i), (ax_i, ay_pred_i, ay_true_i) in zip(amps, axes):
        ax_i.scatter(np.arange(wave_size), x_i)
        ay_pred_i.scatter(np.arange(wave_size), y_pred_i)
        ay_true_i.scatter(np.arange(wave_size), y_true_i)

    plt.tight_layout()
    plt.show()
