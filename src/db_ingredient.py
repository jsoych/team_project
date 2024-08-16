import pandas as pd
from sacred import Ingredient
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


load_dotenv()
DB_URL = os.getenv("DB_URL")

"""
Ingredients are a way of defining a configuration with associated functions
and possibly commands that can be resued by many different experiments.
"""
db_ingredient = Ingredient("db_ingredient")

@db_ingredient.config
def cfg():
    db_url = DB_URL

@db_ingredient.capture
def df_to_sql(df, table_name, db_url=DB_URL, schema=None, if_exists="append"):
    """Convenience function to interface with db."""
    engine = create_engine(db_url)
    with engine.connect() as con:
        df.to_sql(
            table_name,
            con=con,
            if_exists=if_exists,
            schema=schema,
            index=False
        )

