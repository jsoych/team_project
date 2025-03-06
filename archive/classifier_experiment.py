import os
import pandas as pd
from dotenv import load_dotenv

from db_ingredient import df_to_sql
from logger import get_logger
from sacred import Experiment
from sacred.observers import SqlObserver
from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


# Load environment variables
load_dotenv()
db_url = os.getenv("DB_URL")

# Get logger for tracking
_logs = get_logger(__name__)

ex = Experiment("classifier_experiment")
ex.logger = _logs
ex.observers.append(SqlObserver(db_url))

@ex.config
def cfg():
    """
    Defines the needed configuration variables for our experiment.
    """
    experiment_name = "classifier_experiement"
    file = None
    n_splits = 5
    model = None

@ex.capture
def load_data(file):
    """
    Loads the data, and returns response, and predictor variables.
    """
    _logs.info("Load data from {}.".format(file))
    df = pd.read_csv(file)
    y = df.pop("y")
    return df, y

@ex.capture
def evaulate_cross_val(model, X, y, n_splits, experiment_name):
    """
    Evalutates cross validation.
    """
    _logs.info("Evaluate cross validation score")
    k_fold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

    scoring = {
        "acc": accuracy_score,
        "f1": f1_score,
        "log_loss": log_loss,
        "precision": precision_score,
        "recall": recall_score
    }
    # Convert metrics in scoring to scorers
    for name, metric in scoring.items():
        scoring[name] = make_scorer(metric)

    res = cross_validate(
        model, X, y,
        cv=k_fold, scoring=scoring,
        return_train_score=True,
    )
    res = pd.DataFrame(res)

    # meta_df stores experiment specific details for later retrieval 
    meta_df = pd.DataFrame({
        "experiment_name": [experiment_name for _ in range(n_splits)],
        "k_fold_num": [(n+1) for n in range(n_splits)]
        })
    
    return meta_df.join(res)

@ex.capture
def res_to_db(res, table_name):
    _logs.info("Write results to database")
    df_to_sql(res, table_name)

@ex.automain
def run(model):
    _logs.info("Run experiment")
    X, y = load_data()
    res = evaulate_cross_val(model,X,y)
    df_to_sql(res, "cross_validation_results")