from sacred import Ingredient
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.metrics import recall_score, make_scorer
from logger import get_logger

# Create ingredient
ingredient = Ingredient('scorer')
ingredient.logger = get_logger(__name__)

@ingredient.config
def cfg():
    loglevel = 'INFO'

@ingredient.capture
def set_loglevel(loglevel):
    ''' Sets the logger loglevel '''
    ingredient.logger.setLevel(loglevel)

@ingredient.capture
def evaluate(model,X,y,_log):
    ''' 
    Scores the binary classifier againts roc_auc, accuracy, precision and
    recall, and returns the results.
    '''
    metrics = {
        'roc_auc': roc_auc_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    }

    results = {}
    for name, metric in metrics.items():
        results[name] = [make_scorer(metric)(model,X,y)]
    
    return results
