import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dotenv import load_dotenv
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Define the path to the SQLite database file
sql_data_dir = os.getenv("SQL_DATA_DIR")
db_path = os.path.join(sql_data_dir, 'database.db')

# Create a connection to the SQLite3 database
conn = sqlite3.connect(db_path)

# Query data from the database
train_df_3 = pd.read_sql_query("SELECT * FROM train_data", conn)
print(train_df_3.head())

test_df_3 = pd.read_sql_query("SELECT * FROM test_data", conn)
print(test_df_3.head())

# Close the database connection
conn.close()

# Prepare training data
X_3 = train_df_3.iloc[:, :-1].values
Y_3 = train_df_3.iloc[:, -1].values

# Prepare test data
X_test_3 = test_df_3.iloc[:, :-1].values
Y_test_3 = test_df_3.iloc[:, -1].values

# Feature scaling
scaler = StandardScaler()
X_3 = scaler.fit_transform(X_3)
X_test_3 = scaler.transform(X_test_3)

# Add constant to feature matrices
X_3 = sm.add_constant(X_3)
X_test_3 = sm.add_constant(X_test_3)

# Fit the logistic regression model
logit_mod_3 = sm.Logit(Y_3, X_3)
logit_res_3 = logit_mod_3.fit(method="bfgs", maxiter=200, skip_hessian=True)

# Predict probabilities for test data
Y_pred_3 = logit_res_3.predict(X_test_3)

# Plot ROC curve
def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Calculate ROC curve/AUC
fpr_3, tpr_3, thresholds_3 = metrics.roc_curve(Y_test_3, Y_pred_3, pos_label=1)
roc_auc_3 = metrics.auc(fpr_3, tpr_3)

# Plot ROC curve
plot_roc(fpr_3, tpr_3, roc_auc_3)