import pandas as pd
import tensorflow as tf
import keras

from benchmark_model import BenchmarkModel

df = pd.read_csv('/home/jsoychak/cs/ecg/data/mitbih_train.csv')

def prepare_data(data: pd.DataFrame):
    x = data.to_numpy()
    x, y = x[:, :-1], x[:, -1]
    x = tf.expand_dims(x, axis=-1)

    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, 5)
    
    return x, y

x_train, y_train = prepare_data(df)

def compute_class_weights(y_onehot: tf.Tensor):
    total_samples = tf.cast(tf.shape(y_onehot)[0], tf.float32)
    num_classes = tf.cast(tf.shape(y_onehot)[1], tf.float32)

    counts = tf.reduce_sum(y_onehot, axis=0)
    counts = tf.cast(counts, dtype=tf.float32)

    weights = (total_samples / counts) / num_classes
    class_weights = {i: w.numpy() for i, w in enumerate(weights)}
    
    return class_weights

class_weights = compute_class_weights(y_train)

model = BenchmarkModel(32, 5)
model.summary()

epochs = 32
batch_size = 16

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    class_weight=class_weights
)

df_test = pd.read_csv('/home/jsoychak/cs/ecg/data/mitbih_test.csv')

x_test, y_test = prepare_data(df_test)
model.evaluate(x_test, y_test)

model.save('benchmark.keras')
