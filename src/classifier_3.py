import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dotenv import load_dotenv
from sklearn import metrics


load_dotenv()
data_dir = os.getenv("PROCESSED_DATA_DIR")

train_df_3 = pd.read_csv(os.path.join(data_dir, "data_3/train_data.csv"))
print(train_df_3.head())

test_df_3 = pd.read_csv(os.path.join(data_dir, "data_3/test_data.csv"))
print(test_df_3.head())

# Prepare training data
train_data_3 = train_df_3.to_numpy()
Y_3 = train_data_3[:, -1]
X_3 = train_data_3[:, :-1]

# Prepare test data 
test_data_3 = test_df_3.to_numpy()
Y_test_3 = test_data_3[:, -1]
X_test_3 = test_data_3[:, :-1]

# Fit the logistic regression model
logit_mod_3 = sm.Logit(Y_3, X_3)
logit_res_3 = logit_mod_3.fit(method="bfgs", maxiter=100, skip_hessian=True)

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
