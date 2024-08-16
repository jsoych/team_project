import os
import glob
import random
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(tf.keras.utils.PyDataset):

    """
    DataGenerator generates and preprocesses batches of data from the data
    directory. DataGenerator assumes the data directory is structure as 
    follows

    dir_path/
        |-- label_0/
            |-- img0.jpeg
            |-- img1.jpeg
                ...
        |-- label_1/
            |-- img0.jpeg
            |-- img1.jpeg
                ...
            ...
    """
    
    def __init__(
            self,
            dir_path,
            data='auto',
            input_shape=(224,224),
            batch_size=32,
            shuffle=True,
            seed=0,
            preproc_func=None,
            encoder='one_hot',
            **krwags
        ):
        """
        Args:
            dir_path: The path to our data.
            data: If data is set to auto, the data is prepared automatically
                from the structure of the data directory. Otherwise, data is
                is a list of pairs of image paths and labels.
            input_shape: The shape of the images in each batch. If the shape
                of the image is not equal to the specified input shape, a
                copy of the image is resized, then added to the batch.
            batch_size: The size of each batch of data. If the number of
                paths is not a multiple of the batch size, the last batch is
                is smaller.
            shuffle: If True, shuffles the data with seed value.
            seed: Seed value used to shuffle data.
            preproc_func: An optional function to use for preprocessing
                batches of data.
            encoder: If encoder is set to one_hot, a one hot encoding scheme
                is fitted with the labels found in the data directory.
                Otherwise, an object encoder.
        """
        super().__init__(**krwags)
        self.dir_path = dir_path

        # Get labels and label data
        if (data == 'auto'):
            self.labels = self.get_labels('auto')
            self.data = self.__label_data()
        else:
            self.labels = self.get_labels(data)
            self.data = data

        self.input_shape = input_shape
        self.batch_size = batch_size

        # Shuffle data
        if shuffle:
            self.seed = seed
            random.seed(seed)
            random.shuffle(self.data)
        
        self.preproc_func = preproc_func

        # Fit encoder with labels
        if (encoder == 'one_hot'):
            self.encoder = OneHotEncoder(sparse_output=False)
            self.encoder.fit(np.expand_dims(self.labels, axis=-1))
        else:
            self.encoder = encoder

    def __len__(self):
        """ Returns the number of batches. """
        return np.ceil(len(self.data) / self.batch_size).astype('int32')
    
    def __getitem__(self, idx):
        """ Gets the idx'th batch of data. """
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.data))
        x_batch, y_batch = np.array(self.data[low:high]).T

        # Load images
        imgs = []
        for img_path in x_batch:
            with Image.open(img_path) as img:
                # Convert greyscale to RGB and resize image
                if (img.mode == 'L'):
                    img = img.convert('RGB')
                imgs.append(np.array(img.resize(self.input_shape)))
        x_batch = np.array(imgs)

        # Preprocess images
        if (self.preproc_func):
            x_batch = (self.preproc_func(x_batch))

        # Encode labels
        y_batch = (self.encoder.transform(np.expand_dims(y_batch, axis=-1)))

        return x_batch, y_batch
    
    def get_labels(self, data=None):
        """ Get the labels. """
        if (data == 'auto'):
            labels = []
            # Scan data directory for labels
            it = os.scandir(self.dir_path)
            for dir_entry in it:
                if dir_entry.is_dir():
                    labels.append(dir_entry.name)
            it.close()
            return labels
        elif (data):
            labels = set()
            for x, y in data:
                labels.add(y)
            return list(labels)
        else:
            return self.labels


    def __label_data(self):
        """ Labels each image path in the data directory. """
        data = []
        for label in self.labels:
            paths = glob.glob(os.path.join(self.dir_path, label, "*.jpeg"))
            data += [(path, label) for path in paths]
        return data

    def get_paths(self):
        """ Gets all the paths from data. """
        return [path for (path, label) in self.data]
    
    def partition_data_generator(self, frac=0.2, shuffle=True, seed=0):
        """
        Removes frac of the data from the data generator, and returns a 
        new DataGenerator with the same attributes and the removed data. 
        The data that is removed preserves the distribution of labels. 
        That is, if there is 90 data points with label0, 10 data points
        with label1, and frac = 0.1, then partition_data_generator removes
        10 data points with label0, and 1 data point with label1.
        """
        # Store the indexes of each label
        stratified_data = {label: [] for label in self.labels}
        for i, (x,y) in enumerate(self.data):
            stratified_data[y].append(i)
        
        # Sample, and remove frac 
        sample_data = []
        for v in stratified_data.values():
            num_samples = int(frac * len(v))
            for i in v[:num_samples]:
                sample_data.append(self.data.pop(i))
        
        return DataGenerator(
            self.dir_path,
            sample_data,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=seed,
            preproc_func=self.preproc_func,
            encoder=self.encoder
        )
    
    def summary(self):
        """ Prints a summary of all of the data in the data generator. """
        summary = f"data directory: {self.dir_path}\n"
        summary += f"class labels: {self.labels}\n"
        summary += f"number of data points: {len(self.data)}\n"
        
        # Stratify the data by its label
        stratified_data = {label: [] for label in self.labels}
        for x, y in self.data:
            stratified_data[y].append(x)
        
        # Add the size of each class to summary
        for label, data in stratified_data.items():
            summary += f"number of data points in class {label}: {len(data)}\n"

        print(summary)