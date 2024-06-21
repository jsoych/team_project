import numpy as np
import pandas as pd


"""
Everyone make sure they run this scrip from the TEAM_PROJECT directory,
otherwise pd.read_csv will not be able to find the data :(
"""
train_df = pd.read_csv("data/processed/data_1/train_data_1.csv")
print(train_df.head())

test_df = pd.read_csv("data/processed/data_1/test_data_1.csv")
print(test_df.head())

# the following separates our training data (train_df) into response and
# explanatory variables, X and Y respectively
train_data = (train_df.to_numpy()).transpose()
X = train_data[:-1]
Y = train_data[-1]
