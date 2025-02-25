import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from dotenv import load_dotenv
from sklearn import metrics


load_dotenv()
data_dir = os.getenv("PROCESSED_DATA_DIR")

train_df = pd.read_csv(os.path.join(data_dir, "data_1/train_data.csv"))
print(train_df.head())

test_df = pd.read_csv(os.path.join(data_dir, "data_1/test_data.csv"))
print(test_df.head())

# Prepare training data
train_data = train_df.to_numpy()
Y = train_data[:,-1]
X = train_data[:,:-1]
X = sm.tools.tools.add_constant(X)

# Prepare test data
test_data = test_df.to_numpy()
Y_test = test_data[:,-1]
X_test = test_data[:,:-1]
X_test = sm.tools.tools.add_constant(X_test)

# Fit the logistic model
logit_mod = sm.Logit(Y,X)
logit_res = logit_mod.fit(method="bfgs", maxiter=100, skip_hessian=True)

# Predict probabilities for test data
Y_pred = logit_mod.predict(logit_res.params,X_test)

# Plot ROC curve
def plot_roc(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Calculate ROC curve/AUC
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC curve
plot_roc(fpr,tpr,roc_auc)
