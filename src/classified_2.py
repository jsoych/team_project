import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics


train_df_2 = pd.read_csv("data/processed/data_2/train_data_2.csv")
print(train_df_2.head())

test_df_2 = pd.read_csv("data/processed/data_2/test_data_2.csv")
print(test_df_2.head())

# Prepare training data
train_data_2 = train_df_2.to_numpy()
Y_2 = train_data_2[:, -1]
X_2 = train_data_2[:, :-1]

# Prepare test data 
test_data_2 = test_df_2.to_numpy()
Y_test_2 = test_data_2[:, -1]
X_test_2 = test_data_2[:, :-1]

# Fit the logistic regression model
logit_mod_2 = sm.Logit(Y_2, X_2)
logit_res_2 = logit_mod_2.fit(method="bfgs", maxiter=100, skip_hessian=True)

# Predict probabilities for test data
Y_pred_2 = logit_res_2.predict(X_test_2)

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
fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(Y_test_2, Y_pred_2, pos_label=1)
roc_auc_2 = metrics.auc(fpr_2, tpr_2)

# Plot ROC curve
plot_roc(fpr_2, tpr_2, roc_auc_2)
