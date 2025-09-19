from logging import Logger

def get_model(name, logger:Logger=None):
    ''' Gets the model by its name and returns it. '''

    # Log name
    if (logger):
        logger.debug(f'model name: {name}')

    # Get model
    if (name == 'log_reg'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif (name == 'lda'):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif (name == 'qda'):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()  
    elif (name == 'nb'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif (name == 'knn'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    elif (name == 'svm'):
        from sklearn.svm import SVC
        model = SVC()
    
    return model

def get_preproc_func(name, logger:Logger=None):
    ''' Gets the preprocessing function by its name and returns it. '''

    # Log name
    if (logger):
        logger.debug(f'preprocessing function name: {name}')

    # Get preprocessing function
    if (name == 'standard'):
        from sklearn.preprocessing import StandardScaler
        preproc_func = StandardScaler()
    elif (name == 'min_max'):
        from sklearn.preprocessing import MinMaxScaler
        preproc_func = MinMaxScaler()
    elif (name == 'power'):
        from sklearn.preprocessing import PowerTransformer
        preproc_func = PowerTransformer()
    elif (name == 'quantile'):
        from sklearn.preprocessing import QuantileTransformer
        preproc_func = QuantileTransformer()
    
    return preproc_func

def build_model(config:dict, logger:Logger=None):
    ''' Builds and returns the model. '''

    # Log config
    if (logger):
        logger.debug(f'model config: {config}')

    # Get model
    model = get_model(config['name'], logger)

    # Build model
    if ('preprocFunc' in config.keys()):
        # Get preprocessing function
        preproc_func = get_preproc_func(config['preprocFunc'], logger)
        from sklearn.pipeline import Pipeline
        model = Pipeline([('preproc_func', preproc_func), ('model', model)])

    # Set model parameters
    if ('params' in config.keys()):
        model.set_params(**config['params'])
    elif ('paramGrid' in config.keys()):
        from sklearn.model_selection import GridSearchCV
        model = GridSearchCV(model, config['paramGrid'])
    
    return model
