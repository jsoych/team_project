import os
from dotenv import load_dotenv
from classifier_experiment import ex
from sklearn.tree import DecisionTreeClassifier

load_dotenv()
data_dir = os.getenv("PROCESSED_DATA_DIR")

# Create decision tree classifier
tree_classifier = DecisionTreeClassifier(random_state=0)

ex.run(config_updates={
    "experiment_name": "tree_classifier_data_1",
    "file": os.path.join(data_dir,"data_1/train_data.csv"),
    "model": tree_classifier
    }
)

ex.run(config_updates={
    "experiment_name": "tree_classifier_data_2",
    "file": os.path.join(data_dir,"data_2/train_data.csv"),
    "model": tree_classifier
    }
)

ex.run(config_updates={
    "experiment_name": "tree_classifier_data_3",
    "file": os.path.join(data_dir,"data_3/train_data.csv"),
    "model": tree_classifier
    }
)