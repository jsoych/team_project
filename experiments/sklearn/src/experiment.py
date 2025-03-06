import os
import json
import pandas as pd
import database
import models
import scorer

from hashlib import sha256
from logger import get_logger
from sacred import Experiment

# Create experiment
ex = Experiment(
    name='pneumonia_classifier',
    ingredients=[database.ingredient,models.ingredient,scorer.ingredient]
)
ex.logger = get_logger(__name__)

@ex.config
def cfg():
    name = 'logistic_regression'
    model_name = 'logistic_regression'
    train_path = '/data/train_data.csv'
    test_path = '/data/test_data.csv'
    models_table = 'sklearn'
    registry_url = os.getenv('REGISTRY_URL')
    results_table = 'results'
    results_url = os.getenv('RESULTS_URL')

@ex.capture
def load_data(path, seed, sample=False, frac=None):
    ''' Loads data from path, and returns X and y. '''
    df = pd.read_csv(path)
    if sample:
        df = df.sample(frac=frac,random_state=seed)
    y = df.pop('y')
    return df.to_numpy(), y.to_numpy()

@ex.capture
def save_results(results, id, name, results_table, results_url, _log) -> None:
    ''' Saves the results to the database If the url is None, save_results
        returns None. '''
    if (results_url == None):
        return None
    
    results['id'] = id
    results['name'] = name
    results_df = pd.DataFrame(
        results,
        index=None,
        columns=['id','name','roc_auc','accuracy','precision','recall']
    )
    database.df_to_sql(results_df,results_table,results_url)

@ex.capture
def save_model(model, id, name, models_table, registry_url, _log) -> None:
    ''' Saves the model to the registry. If the url is None, save_model 
        returns None. '''
    if (registry_url == None):
        return None
    
    df = pd.DataFrame(
        {
            'id': [id],
            'name': [name],
            'model': [models.serialize_model(model)]
        },
        index=None,
        columns=['id','name','model']
    )
    database.df_to_sql(df,models_table,registry_url)

@ex.automain
def main(model_name, train_path, test_path, _log, _run):
    ''' Run the experiment. '''
    _log.info(f'Runinng the experiment')

    # Loading training data
    _log.info(f'Loading training data')
    _log.debug(f'train_path: {train_path}')
    X, y = load_data(train_path)
    _log.debug(f'X.shape: {X.shape}')
    _log.debug(f'y.shape: {y.shape}')

    # Getting the model
    _log.info(f'Getting model')
    _log.debug(f'model name: {model_name}')
    model = models.get_model(model_name)

    # Fitting the model
    _log.info(f'Fitting model')
    model.fit(X,y)

    # Loading test data
    _log.info(f'Loading test data')
    _log.debug(f'test_path: {test_path}')
    X_test, y_test = load_data(test_path)
    _log.debug(f'X_test.shape: {X_test.shape}')
    _log.debug(f'y_test.shape: {y_test.shape}')

    # Evaluating the model
    _log.info(f'Evaluating the model')
    results = scorer.evaluate(model,X_test,y_test)
    _log.debug(f'results: {results}')

    # Create experiement id
    config = json.dumps(_run.config)
    id = sha256(config.encode()).hexdigest()
    _log.debug(f'config: {config}')
    _log.debug(f'id: {id}')
    
    # Save results to results database
    _log.info('Saving experiment results to database')
    save_results(results,id)

    # Save the model to the model registry
    _log.info('Saving model to registry')
    save_model(model,id)