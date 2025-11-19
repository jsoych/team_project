import keras

@keras.saving.register_keras_serializable()
class ConvBlock1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, name='conv_block', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.name_prefix = name
        
        self.conv_layer_1 = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding='same',
            activation='relu',
            name=f'{name}_conv1'
        )
        self.conv_layer_2 = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding='same',
            name=f'{name}_conv2'
        )
    
    def call(self, inputs):
        x = self.conv_layer_1(inputs)
        return self.conv_layer_2(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'name': self.name_prefix
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class DeepCNN(keras.Model):
    def __init__(self, filters, kernel_size, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth

        self.conv_input = keras.layers.Conv1D(filters, kernel_size, padding='same', name='conv_input')
        self.conv_blocks = [
            ConvBlock1D(filters, kernel_size, name=f'conv_block{i+1}') for i in range(depth)
        ]
        self.relu = keras.activations.relu
        self.pool = keras.layers.MaxPool1D(kernel_size, strides=2, name='max_pooling')
        self.flatten = keras.layers.Flatten()
        self.hidden_dense = keras.layers.Dense(filters, activation='relu', name='hidden_dense')
        self.classifier = keras.layers.Dense(5, activation='softmax', name='classifier')

    def call(self, inputs):
        x = self.conv_input(inputs)
        for block in self.conv_blocks:
            x = block(x)
            x = self.relu(x)
            x = self.pool(x)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'depth': self.depth
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class ResNet(keras.Model):
    def __init__(self, filters, kernel_size, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.conv_input = keras.layers.Conv1D(filters, kernel_size, padding='same', name='conv_input')
        self.conv_blocks = [
            ConvBlock1D(filters, kernel_size, name=f'conv_block{i + 1}') for i in range(depth)
        ]
        self.relu = keras.activations.relu
        self.pooling = keras.layers.MaxPool1D(kernel_size, strides=2, name='max_pooling')
        self.flatten = keras.layers.Flatten()
        self.hidden_dense = keras.layers.Dense(filters, activation='relu', name='hidden_dense')
        self.classifier = keras.layers.Dense(5, activation='softmax', name='classifier')
    
    def call(self, inputs):
        x = self.conv_input(inputs)
        for block in self.conv_blocks:
            w = block(x)
            x = self.relu(x + w)
            x = self.pooling(x)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'depth': self.depth
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
