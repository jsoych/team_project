import os
from dotenv import load_dotenv
from classifier_experiment import ex
from sklearn.linear_model import LogisticRegression


load_dotenv()
data_dir = os.getenv("PROCESSED_DATA_DIR")

# Create logistic regression model
log_reg = LogisticRegression(max_iter=1000)

ex.run(config_updates={
    "experiment_name": "log_reg_data_1",
    "file": os.path.join(data_dir,"data_1/train_data.csv"),
    "model": log_reg
    }
)
ex.run(config_updates={
    "experiment_name": "log_reg_data_2",
    "file": os.path.join(data_dir,"data_2/train_data.csv"),
    "model": log_reg
    }
)
ex.run(config_updates={
    "experiment_name": "log_reg_data_3",
    "file": os.path.join(data_dir,"data_3/train_data.csv"),
    "model": log_reg
    }
)