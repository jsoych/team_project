from pandas import DataFrame
from sacred import Ingredient
from sqlalchemy import create_engine
from get_logger import get_logger

ingredient = Ingredient("database")
ingredient.logger = get_logger(__name__) 

@ingredient.config
def cfg():
    loglevel = 'INFO'

@ingredient.capture
def set_loglevel(loglevel):
    ''' Set the logger loglevel '''
    ingredient.logger.setLevel(loglevel)

@ingredient.capture
def df_to_sql(df: DataFrame, table_name, db_url, _log, schema=None, if_exists="append"):
    ''' Inserts dataframe into database. '''
    _log.info(f'Inserting dataframe to {table_name}')
    _log.debug(f'Trying to connect to database with {db_url}')
    engine = create_engine(db_url)
    with engine.connect() as con:
        df.to_sql(table_name, con, schema, if_exists, index=False)
