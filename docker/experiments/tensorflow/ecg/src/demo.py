import numpy as np
import pandas as pd
import tensorflow as tf

from ecg_dataset import EcgDataset, plot_batch
from wave_transformer import WaveTransformer, plot_results, plot_wave_transform

def run():
    df = pd.read_csv('/home/jsoychak/cs/ecg/data/mitbih_train.csv')
    df.describe()

    data = df.to_numpy()
    X = data[:-1]
    y = data[-1]

    X = np.expand_dims(X, axis=-1)

    # parameters
    epochs = 2
    batch_size = 16
    wave_size = 64
    window_size =  96
    hidden_units = 256

    dataset = EcgDataset(X, batch_size, wave_size, window_size)

    model = WaveTransformer(window_size, wave_size, hidden_units)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )

    history = model.fit(dataset, epochs=epochs)

    # plot a batch and its results
    x, y = dataset.__getitem__(0)
    sample_size = 4

    plot_batch(x[:sample_size], y[:sample_size])

    plot_results(model, x[:sample_size], y[:sample_size])

    plot_wave_transform(model, x[:sample_size], y[:sample_size])

    # compare against randomly initialized transformer
    random_model = WaveTransformer(window_size, wave_size, hidden_units)

    plot_wave_transform(random_model, x[:sample_size], y[:sample_size])

if __name__ == '__main__':
    run()
