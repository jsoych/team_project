import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

from sklearn import metrics


"""
Everyone make sure they run this script from the TEAM_PROJECT directory,
otherwise pd.read_csv will not be able to find the data :(
"""
train_df = pd.read_csv("data/processed/data_1/train_data_1.csv")
print(train_df.head())

test_df = pd.read_csv("data/processed/data_1/test_data_1.csv")
print(test_df.head())

# the following separates our training data (train_df) into response and
# explanatory variables, Y and X respectively
train_data = train_df.to_numpy()
Y = train_data[:,-1]
X = train_data[:,:-1]

# the following defines and fits the logistic regression model with our
# training data
logit_mod = sm.Logit(Y,X)
logit_res = logit_mod.fit(
    method="bfgs",
    maxiter=100,
    skip_hessian=True
)

# the following separates our test data into reponse and explantory
# variables, Y_test and X_test respectively
test_data = test_df.to_numpy()
Y_test = test_data[:,-1]
X_test = test_data[:,:-1]

# the following scores our model againts the test data
Y_pred = logit_mod.predict(logit_res.params,X_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

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
    return

plot_roc(fpr,tpr,roc_auc)