Ran the python script for logistic regression and got the following:


precision    recall  f1-score   support

           0       0.96      0.53      0.69       234
           1       0.78      0.99      0.87       390

    accuracy                           0.82       624
   macro avg       0.87      0.76      0.78       624
weighted avg       0.85      0.82      0.80       624

AUC-ROC: 0.9367959675651983

![Confusion_Matrix](https://github.com/jsoych/team_project/assets/35706356/b838b06d-f30d-45ec-bba3-ad1509034d20)

![ROC_Curve](https://github.com/jsoych/team_project/assets/35706356/51d05049-5b57-43b0-83fb-830897ee81f2)

Here's an interpretation of your logistic regression model's performance:

- **Precision**: For class 0 (negative class), the precision is **0.96**, meaning that when the model predicts class 0, it is correct 96% of the time. For class 1 (positive class), the precision is **0.78**, so it's correct 78% of the time when predicting class 1.

- **Recall**: The recall for class 0 is **0.53**, indicating that the model correctly identifies 53% of all actual class 0 instances. For class 1, the recall is **0.99**, which is excellent as it identifies 99% of all actual class 1 instances.

- **F1-Score**: The F1-score for class 0 is **0.69**, and for class 1, it's **0.87**. The F1-score is a harmonic mean of precision and recall, providing a balance between the two. A higher F1-score indicates a better model.

- **Support**: This is the number of actual occurrences of each class in your dataset. There are **234** instances of class 0 and **390** instances of class 1.

- **Accuracy**: The overall accuracy of the model is **0.82**, meaning it correctly predicts the outcome 82% of the time across both classes.

- **Macro Avg**: The macro average for precision, recall, and F1-score is **0.87**, **0.76**, and **0.78** respectively, which averages the performance across classes without taking class imbalance into account.

- **Weighted Avg**: The weighted average for precision, recall, and F1-score is **0.85**, **0.82**, and **0.80** respectively, which accounts for class imbalance by weighting the average based on the support of each class.

- **AUC-ROC**: The AUC-ROC score is **0.937**, which is close to 1, indicating that the model has a high level of discrimination ability between the positive and negative classes.

- **Confusion Matrix**: The confusion matrix shows that out of **234** actual class 0 instances, **125** were correctly predicted as class 0, and **109** were incorrectly predicted as class 1. For class 1, out of **390** actual instances, **385** were correctly predicted, and only **5** were incorrectly predicted as class 0.

Overall, the model shows a strong ability to identify class 1 instances but seems to struggle with class 0, as indicated by the lower recall for class 0. The high AUC-ROC score suggests that the model's ability to distinguish between the classes is good. However, further investigation why the model has a lower recall for class 0 and consider strategies to improve it, such as resampling, feature engineering, or tuning the decision threshold.
