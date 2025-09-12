import os
import logging
import logging.config
import models
import yaml

from data_generator import DataGenerator
from hashlib import sha256
from pandas import DataFrame
from sqlalchemy import create_engine

# Configure logger
with open(os.getenv('LOGGER_CONFIG', default='/mnt/configs/logger.yaml')) as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)

# Create logger
_log = logging.getLogger(__name__)
_log.setLevel(os.getenv('LOG_LEVEL', default='INFO'))

# Load experiment configuration
with open(os.getenv('EXPERIMENT_CONFIG', default='../configs/experiment.yaml'), 'rb') as f:
    config = yaml.safe_load(f)
    _log.debug(config)

# Create train, val, and test data generators
_log.info('Creating the train data generator')
train_gen = DataGenerator(
    os.getenv('TRAIN_DIR', default='/mnt/data/chest_xray/train'),
    batch_size=config['batch_size']
)
train_gen.preproc_func = models.get_preproc_func(config['model']['preproc_func'], _log)
train_gen.summary()

_log.info('Creating the val data generator')
val_gen = DataGenerator(
    os.getenv('VAL_DIR', default='/mnt/data/chest_xray/val'),
    batch_size=config['batch_size']
)
val_gen.preproc_func = models.get_preproc_func(config['model']['preproc_func'], _log)
val_gen.summary()

_log.info('Creating the test data generator')
test_gen = DataGenerator(
    os.getenv('TEST_DIR', default='/mnt/data/chest_xray/test'),
    batch_size=config['batch_size']
)
test_gen.preproc_func = models.get_preproc_func(config['model']['preproc_func'], _log)
test_gen.summary()

# Create model
_log.info('Creating the model')
model = models.model_builder(config['model']['name'], config['model']['arch'], _log)
model.summary()

def df_to_sql(df: DataFrame, table_name, db_url):
    ''' Inserts the dataframe into the database. '''
    _log.info(f'Inserting dataframe into {table_name}')
    _log.debug(f'Trying to connect to database with {db_url}')
    engine = create_engine(db_url)
    with engine.connect() as con:
        df.to_sql(table_name, con, if_exists='append', index=False)
    return None

def create_experiment_id(config):
    ''' Creates a unique id from the experiment configuration '''
    h = sha256()
    h.update(config['name'].encode())
    for l in config['model']['arch']['layers']:
        h.update(l['name'].encode())
    h.update(str(config['epochs']).encode())
    h.update(config['registry_table'].encode())
    h.update(config['results_table'].encode())
    return h.hexdigest()

def save_results(results, experiment_id, experiment_name, table_name):
    ''' Saves the results to the database. If the RESULTS_URL is none,
        save_results returns None. '''
    url = os.getenv('RESULTS_URL')
    if url == None:
        return None
    data = {k: [v] for k,v in results.items()}
    data['id'] = [experiment_id]
    data['name'] = [experiment_name]
    results_df = DataFrame(
        data=data,
        index=None,
        columns=['id', 'name', 'roc_auc', 'accuracy'],
    )
    df_to_sql(results_df, table_name, url)
    return None

def save_model(model, experiment_id, experiment_name, table_name):
    ''' Saves the model to the registry. If the REGISTRY_URL is none,
        save_model returns None. '''
    url = os.getenv('REGISTRY_URL')
    if url == None:
        return None
    path = os.path.join(os.getenv('REGISTRY_DIR', default='/mnt/registry/tensorflow'), experiment_id + '.keras')
    _log.debug(path)
    model.save(path)
    model_df = DataFrame(
        data={'id': [experiment_id], 'name': [experiment_name], 'path': [path]},
        index=None,
        columns=['id', 'name', 'path']
    )
    df_to_sql(model_df, table_name, url)
    return None

def main():
    ''' Run the experiment '''
    _log.info('Running experiment')

    # Create experiment id
    experiment_id = create_experiment_id(config)
    _log.debug(f'experiment_id: {experiment_id}')
    
    # Fit the model
    _log.info('Fitting the model')
    model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=config['epochs']
    )

    # Evaluate the model
    _log.info('Evaluting the model')
    results = model.evaluate(test_gen, return_dict=True)
    _log.info(f'results: {results}')
    
    # Save results to results database
    _log.info('Saving results to results database')
    save_results(results, experiment_id, config['name'], config['results_table'])

    # Save model to registry database
    _log.info('Saving model to registry database')
    save_model(model, experiment_id, config['name'], config['registry_table'])

if __name__ == '__main__':
    main()
