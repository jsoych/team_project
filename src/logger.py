import logging
import os

from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
log_dir = os.getenv("LOG_DIR")
log_level = os.getenv("LOG_LEVEL")

def get_logger(name, log_dir=log_dir, log_level=log_level):
    """
    Set up a logger with the given name, and log level.
    """
    # Create log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    _logs = logging.getLogger(name)

    # Handlers send the log records to the appropriate destinvation.
    f_handler = logging.FileHandler(os.path.join(log_dir, f'{ datetime.now().strftime("%Y%m%d_%H%M%S") }.log'))

    # Formatters specify the layout of log records in the final output.
    f_format = logging.Formatter('%(asctime)s, %(name)s, %(filename)s, %(lineno)d, %(funcName)s, %(levelname)s, %(message)s')
    f_handler.setFormatter(f_format)

    _logs.addHandler(f_handler)
    _logs.setLevel(log_level)
    return _logs        