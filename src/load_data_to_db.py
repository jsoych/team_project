import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

# Define the path to the data directory
# data_dir = '/home/cyber/Documents/team_project/data/processed/data'

load_dotenv()
processed_data_dir = os.getenv("PROCESSED_DATA_DIR")
data_dir = os.getenv("DATA_DIR")


# Define the paths to the CSV files
train_csv = os.path.join(processed_data_dir, 'data_1/train_data.csv')
test_csv = os.path.join(processed_data_dir, 'data_1/test_data.csv')

# Read the CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Create a connection to the SQLite3 database
conn = sqlite3.connect(os.path.join(data_dir, 'sql/database.db'))

# Store the data in the SQL database
train_df.to_sql('train_data', conn, if_exists='replace', index=False)
test_df.to_sql('test_data', conn, if_exists='replace', index=False)

# Close the database connection
conn.close()

print("Data has been stored in the SQL database.")