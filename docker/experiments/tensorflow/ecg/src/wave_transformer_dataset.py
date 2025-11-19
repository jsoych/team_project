import copy
import keras
import numpy as np
import matplotlib.pyplot as plt

class EcgDataset(keras.utils.PyDataset):
    def __init__(self, X, batch_size, wave_size, window_size, noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.batch_size = batch_size
        self.wave_size = wave_size
        self.window_size = window_size
        self.noise_std = noise_std

        self.X_noisy = copy.deepcopy(X)
        self.X_noisy += np.random.normal(loc=0, scale=noise_std, size=self.X_noisy.shape)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        assert len(self.X[idx]) >= self.window_size
        step_size = (len(self.X[idx]) - self.window_size + 1)//self.batch_size
        steps = range(0, step_size*self.batch_size, step_size)

        x_batch = np.array([
            self.X_noisy[idx][s: s + self.window_size] for s in steps
        ])

        y_batch = np.array([
            self.X[idx][s + (self.window_size - self.wave_size)//2:
                   s + (self.window_size - self.wave_size)//2 + self.wave_size]
            for s in steps
        ])
        return x_batch, y_batch

def plot_batch(x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)

    batch_size, window_size = x.shape
    _, wave_size = y.shape

    _, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 2))
    axes = np.atleast_2d(axes)

    batch = zip(x, y)
    for (x_i, y_i), (ax_x, ay_y) in zip(batch, axes):
        ax_x.plot(np.arange(window_size), x_i)
        ay_y.plot(np.arange(wave_size), y_i)
    
    plt.tight_layout()
    plt.show()
