import math
import tensorflow as tf

class ECGDataset(tf.keras.utils.PyDataset):
    def __init__(self, x, y, batch_size=32, normalize=True, num_classes=5):
        """
        TensorFlow-native ECG dataset sequence.

        Args:
            x: tf.Tensor or np.array, shape (num_samples, signal_length)
            y: tf.Tensor or np.array, shape (num_samples,)
            batch_size: int
            normalize: bool, whether to normalize each batch
            num_classes: int, number of output classes
        """
        self.x = tf.cast(x, tf.float32)
        self.x = tf.expand_dims(self.x, axis=-1)  # Add channel dimension
        self.y = tf.cast(y, tf.int32)
        self.y = tf.one_hot(self.y, depth=num_classes)

        self.batch_size = batch_size

        if normalize:
            mean = tf.reduce_mean(self.x)
            std = tf.math.reduce_std(self.x)
            self.x = (self.x - mean)/std

        self.num_classes = num_classes

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        low = self.batch_size * index
        high = min(self.batch_size * (index + 1), len(self.x))

        x_batch = self.x[low:high]
        y_batch = self.y[low:high]

        return x_batch, y_batch

    def class_weights(self):
        counts = tf.reduce_sum(self.y, axis=0)
        total_samples = tf.cast(tf.shape(self.y)[0], tf.float32)
        num_classes = tf.cast(self.num_classes, tf.float32)
        weights = (total_samples / counts) / num_classes
        return {i: w.numpy() for i, w in enumerate(weights)}

    def summary(self):
        print(f"Total samples: {len(self.x)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {len(self)}")
        print(f"Signal shape: {self.x.shape[1:]}")
        print(f"Number of classes: {self.num_classes}")

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('/home/jsoychak/cs/team_project/data/raw/ecg/mitbih_train.csv')
    ds = ECGDataset(df.to_numpy()[:,:-1], df.to_numpy()[:,-1])
    ds.summary()
