import os
import json
import pandas as pd
import database
import models
import scorer

from hashlib import sha256
from get_logger import get_logger
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
def save_results(results, id, name, results_table, results_url):
    ''' Saves the results to the database '''
    results['id'] = id
    results['name'] = name
    results_df = pd.DataFrame(
        results,
        index=None,
        columns=['id','name','roc_auc','accuracy','precision','recall']
    )
    database.df_to_sql(results_df,results_table,results_url)

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
    _log.info('Saving model results')
    save_results(results,id)
