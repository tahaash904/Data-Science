import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df_train=pd.read_csv('../datasets/train_u6lujuX_CVtuZ9i.csv')
df_test=pd.read_csv('../datasets/test_Y3wMUE5_7gLdaTN.csv')
# print(df_train.shape)
# print(df_test.shape)
# print(df_train.head(10))
# print(df_test.head(10))
print(df_train.isna().sum())
df_train['Gender'] = df_train['Gender'].fillna(df_train['Gender'].mode().values[0] )
df_train['Married'] = df_train['Married'].fillna(df_train['Married'].mode().values[0] )
df_train['Dependents'] = df_train['Dependents'].fillna(df_train['Dependents'].mode().values[0] )
df_train['Self_Employed'] = df_train['Self_Employed'].fillna(df_train['Self_Employed'].mode().values[0] )
df_train['LoanAmount'] = df_train['LoanAmount'].fillna(df_train['LoanAmount'].median() )
df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mode().values[0] )
df_train['Credit_History'] = df_train['Credit_History'].fillna(df_train['Credit_History'].mode().values[0] )
# print(df_train.isna().sum())
# print(df_test.isna().sum())
fig=plt.figure(figsize=(10,6))
# sns.countplot(y='Gender',hue='Loan_Status', data=df_train)
# grid=sns.FacetGrid(df_train,row='Gender',col='Married',size=2.2,aspect=1.6)
# grid.map(plt.hist,'ApplicantIncome',alpha=.5,bins=10)
# grid.add_legend()
# plt.show()
code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}
df_train=df_train.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
df_test=df_test.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
df_train.drop('Loan_ID', axis=1,inplace=True)
# print(df_train['Dependents'].value_counts())
#print(df_train.info())
Dependents_=pd.to_numeric(df_train['Dependents'])
Dependents__=pd.to_numeric(df_test['Dependents'])
df_train.drop(['Dependents'],axis=1,inplace=True)
df_test.drop(['Dependents'],axis=1,inplace=True)
df_train=pd.concat([df_train,Dependents_],axis=1)
df_test=pd.concat([df_test,Dependents__],axis=1)
# print(df_train.info())
# sns.heatmap(df_train.corr())
# plt.show()
y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
evaluation = f1_score(y_test, y_pred)
print(evaluation)