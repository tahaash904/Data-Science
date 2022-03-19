import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv('../datasets/creditcard.csv')
# print(data.describe())
print(data.shape)
# print(data.columns)
data = data.sample(frac=.1,random_state=1)
print(data.shape)
# data.hist(figsize=(10,10))
# plt.show()
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud cases {}".format(len(Fraud)))
print("Valid cases {}".format(len(Valid)))
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = 'Class'
x=data[columns]
y=data[target]
print(x.shape)
print(y.shape)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
state =1
classifiers ={
    "Isolation Forest": IsolationForest(max_samples=len(x),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)

}
n_outliers =len(Fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        y_pred=clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != y).sum()
    print('{} {}'.format(clf_name,n_errors))
    print(accuracy_score(y_pred,y))
    print(classification_report(y,y_pred))