import pickle

from sacred import Ingredient
from logger import get_logger

# Create ingredient
ingredient = Ingredient('models')
ingredient.logger = get_logger(__name__)

@ingredient.config
def cgf():
    loglevel = 'INFO'

@ingredient.capture
def set_loglevel(loglevel):
    ''' Sets the logger loglevel '''
    ingredient.logger.setLevel(loglevel)

@ingredient.capture
def get_model(name,_log):
    ''' Returns the model by name. '''
    _log.debug(f'model_name: {name}')

    if name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

    elif name == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()

    elif name == 'QDA':
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()
    
    elif name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    elif name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()

    return model

def serialize_model(model):
    ''' Serializes the model for storage and returns it. '''
    return pickle.dumps(model)

def deserialize_model(model):
    ''' Deserializes the model and returns it. '''
    return pickle.loads(model)