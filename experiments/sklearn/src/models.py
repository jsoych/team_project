from logging import Logger

def get_model(name, logger: Logger):
    ''' Gets the model by its name and returns it. '''
    logger.debug(f'model name: {name}')
    if (name == 'logistic_regression'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif (name == 'LDA'):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif (name == 'QDA'):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()  
    elif (name == 'NB'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif (name == 'KNN'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    return model

def get_preproc_func(name, logger: Logger):
    ''' Gets the preprocessing function by its name and returns it. '''
    logger.debug(f'preprocessing function name: {name}')
    if (name == 'standard'):
        from sklearn.preprocessing import StandardScaler
        preproc_func = StandardScaler()
    elif (name == 'power'):
        from sklearn.preprocessing import PowerTransformer
        preproc_func = PowerTransformer()
    elif (name == 'quantile'):
        from sklearn.preprocessing import QuantileTransformer
        preproc_func = QuantileTransformer()
    return preproc_func

def build_model(config: dict, logger: Logger):
    ''' Builds and returns the model. '''
    logger.debug(f'model config: {config}')
    model = get_model(config['name'], logger)

    if ('preprocFunc' in config.keys()):
        preproc_func = get_preproc_func(config['preprocFunc'], logger)
        from sklearn.pipeline import Pipeline
        model = Pipeline([('preproc_func', preproc_func), ('model', model)])
    
    if ('params' in config.keys()):
        model.set_params(**config['params'])
    elif ('paramGrid' in config.keys()):
        from sklearn.model_selection import GridSearchCV
        model = GridSearchCV(model, config['paramGrid'])
    
    return model
