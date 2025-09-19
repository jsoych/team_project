import os
import random
import yaml

preproc_funcs = ['none', 'standard', 'min_max', 'power', 'quantile']
models = ['log_reg', 'lda', 'qda', 'nb', 'knn', 'svm']

def make_config(data, func, model, models_table, results_table):
    ''' Makes a experiment configuration and returns it. '''

    # Make experiment configuration
    config = {}
    config['name'] = '_'.join([data, func, model])
    config['train_path'] = '/mnt/data/' + data + '/train_data.csv'
    config['test_path'] = '/mnt/data/' + data + '/test_data.csv'
    config['models_table'] = models_table
    config['results_table'] = results_table

    # Make model configuration
    model_config = {}
    model_config['name'] = model
    if (func != 'none'):
        model_config['preproc_func'] = func
    config['model'] = model_config

    return config

if __name__ == '__main__':
    # Load environment variables
    data = os.getenv('DATA_SET')
    target_dir = os.getenv('TARGET_DIR')

    # Create configuration space
    configs = []
    for func in preproc_funcs:
        for model in models:
            configs.append((func, model))
            
    # Make test configurations
    if (n := os.getenv('TEST_SIZE')):
        # Cast n to int
        n = int(n)

        # Make test configs directory
        test_dir = os.path.join(target_dir, 'test')
        if (not os.path.exists(test_dir)):
            os.mkdir(test_dir)

        # Make test sample
        for _ in range(n):
            func, model = random.choice(configs)
            config = make_config(data, func, model, 'test_models', 'test_results')
            
            with open(os.path.join(test_dir, f'{config['name']}.yaml'), 'w') as file:
                yaml.safe_dump(config, file)
    
    # Make configs directory
    configs_dir = os.path.join(target_dir, data)
    if (not os.path.exists(configs_dir)):
        os.mkdir(configs_dir)

    # Make configs
    for func, model in configs:
        config = make_config(data, func, model, 'sklearn_models', 'sklearn_results')

        with open(os.path.join(configs_dir, f'{config['name']}.yaml'), 'w') as file:
            yaml.safe_dump(config, file)
        
        del config
