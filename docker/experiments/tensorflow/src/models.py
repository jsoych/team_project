import json
import pickle

from keras import Input, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.losses import CategoricalCrossentropy 
from keras.metrics import AUC, CategoricalAccuracy
from keras.optimizers import Adam


def get_layer(name, kwargs, trainable, logger):
    ''' Returns a new layer by name '''
    layer = None
    if name == 'res-net50':
        from keras.applications import ResNet50
        layer = ResNet50(**kwargs)
    elif name == 'vgg19':
        from keras.applications import VGG19
        layer = VGG19(**kwargs)
    elif name == 'xception':
        from keras.applications import Xception
        layer = Xception(**kwargs)
    elif name == 'dense':
        if 'units' not in kwargs.keys():
            _log.info(f'Error: {name} layer is missing units argument')
            return None
        layer = Dense(**kwargs)
    elif name == 'dropout':
        if 'rate' not in kwargs.keys():
            _log.info(f'Error: {name} layer is missing rate argument')
            return None
        layer = Dropout(**kwargs)
    elif name == 'flatten':
        layer = Flatten(**kwargs)
    elif name == 'input':
        layer = Input(**kwargs)
    else:
        logger.info(f'Error: {name} layer is unknown')
        return None
    layer.trainable = trainable
    return layer

def get_preproc_func(name, logger):
    ''' Returns a preprocessing function by name '''
    if name == 'res-net50':
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif name == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
        return preprocess_input
    elif name == 'xception':
        from keras.applications.xception import preprocess_input
        return preprocess_input
    else:
        logger.info(f'Error: {name} preprocessing function is unknown')
    return None

def model_builder(name, arch, logger):
    ''' Builds and returns a model '''
    logger.debug(f'arch: {arch}')
    model = Sequential(name=name)
    for layer in arch['layers']:
        new_layer = get_layer(layer['name'], layer['args'], layer['trainable'], logger)
        if new_layer == None:
            logger.info(f'Error: model_builder unable to build model') 
            return None
        model.add(new_layer)
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[AUC(name='roc_auc'), CategoricalAccuracy(name='accuracy')]
    )
    return model

