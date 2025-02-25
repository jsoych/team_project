import logging
import logging.config
import yaml

# Configure logger
with open('logger_config.yaml') as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)

def get_logger(name,level='INFO'):
    ''' Creates and returns a logger '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger