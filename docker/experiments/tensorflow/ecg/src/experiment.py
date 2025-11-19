import os
import logging
import logging.config
import tensorflow as tf
import pandas as pd
import yaml

from logging import Logger
from hashlib import sha256
from ecg_dataset import ECGDataset
from models import DeepCNN

def create_experiment_id(config:dict, logger:Logger=None):
    ''' Creates an id from the experiment configuration and returns it. '''

    # Log config
    if (logger):
        logger.debug(f'create_experiment: config: {config}')

    # Create unique id
    h = sha256()

    keys = ['name', 'train_path', 'test_path']
    for k in keys:
        h.update(config[k].encode())
    
    h.update(config['model']['name'].encode())

    if ('preprocFunc' in config.keys()):
        h.update(config['preprocFunc'].encode())
    
    return h.hexdigest()[:64]

# Confiure logger
with open(os.getenv('LOGGER_CONFIG', default='/mnt/configs/logger.yaml')) as file:
    logger_config = yaml.safe_load(file)
logging.config.dictConfig(logger_config)

# Create logger
_log = logging.getLogger(__name__)
_log.setLevel(os.getenv('LOG_LEVEL', default='INFO'))

# Load experiment configuration
with open(os.getenv('EXPERIMENT_CONFIG', default='../configs/experiment.yaml')) as file:
    config = yaml.safe_load(file)
    _log.debug(config)

def create_dataset(path, batch_size):
    _log.info(f'Creating data set with batch size {batch_size}')
    _log.info(f'Loading data from {path}')
    df = pd.read_csv(path)
    return ECGDataset(df.to_numpy()[:,:-1], df.to_numpy()[:,-1], batch_size)

def run(config, logger):
    # Load configuration file
    with open('../configs/experiment.yaml') as file:
        config = yaml.safe_load(file)

    # Load data
    df = pd.read_csv('/home/jsoychak/cs/ecg/data/mitbih_train.csv')

    # Create dataset
    dataset = ECGDataset(
        x=df.to_numpy()[:64,:-1],
        y=df.to_numpy()[:64,-1], 
        batch_size=config['experiment']['batch_size']
    )
    dataset.summary()

    # Create model
    model = DeepCNN(**config['model'])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )

    model.fit(
        dataset,
        epochs=config['experiment']['epochs'],
        class_weight=dataset.class_weights()
    )

    model.save('deep_cnn.keras')

if __name__ == '__main__':
    run()
