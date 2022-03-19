import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('../datasets/heart.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
# print(dataset.head(6))
# print(dataset.shape)
# missing=dataset.isnull().sum()
# percent_1=dataset.isnull().sum()/dataset.isnull().count()
# percent_2=round(percent_1,1)
# missing_data=pd.concat([missing,percent_2],axis=1,keys=['total','%'])
# print(missing_data)
# fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10, 4))
# man=dataset[dataset['sex']==1]
# female=dataset[dataset['sex']==0]
# ax=sns.distplot(female[female['target']==1].age.dropna(),bins=18,label='Have disease',ax=axes[0],kde=False)
# ax=sns.distplot(female[female['target']==0].age.dropna(),bins=18,label='Not disease',ax=axes[0],kde=False)
# ax.legend()
# ax.set_title('female')
# ax = sns.distplot(man[man['target']==1].age.dropna(), bins=18, label = 'Have disease', ax = axes[1], kde = False)
# ax = sns.distplot(man[man['target']==0].age.dropna(), bins=18, label = 'Not disease', ax = axes[1], kde = False)
# ax.legend()
# ax.set_title('Male')
# ax=sns.distplot(dataset[dataset['target']==1].age.dropna(),bins=18,label='Have disease',ax=axes,kde=False)
# ax=sns.distplot(dataset[dataset['target']==0].age.dropna(),bins=18,label='Not disease',ax=axes,kde=False)
# ax.legend()
# ax=sns.boxplot(palette='Set2',orient='h',data=dataset[dataset['target']==1])
# plt.show()
dataset['oldpeak']=dataset['oldpeak'].astype(int)
dataset=pd.concat([dataset,pd.get_dummies(dataset['restecg'],prefix='restecg')],axis=1)
dataset=pd.concat([dataset,pd.get_dummies(dataset['cp'],prefix='cp')],axis=1)
dataset=pd.concat([dataset,pd.get_dummies(dataset['thal'],prefix='thal')],axis=1)
dataset=pd.concat([dataset,pd.get_dummies(dataset['ca'],prefix='ca')],axis=1)
dataset=pd.concat([dataset,pd.get_dummies(dataset['slope'],prefix='slope')],axis=1)
dataset=dataset.drop(['restecg','cp','thal','ca','slope'],axis=1)
x=dataset.drop(['target'],axis=1)
y=dataset['target']
# print(x.head(6))
# print(x.describe())
# print(x.info())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
# print(x_train.shape)
# print(x_test.shape)
corrm=x_train.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corrm, annot=True,ax=ax)
# plt.show()
corrMat=set()
# for i in range(len(corrm.columns)):
#     for j in range(i):
#         if abs(corrm.iloc[i, j]) > 0.85:
#             colname = corrm.columns[i]
#             corrMat.add(colname)
# print(corrMat)
x_train=x_train.drop(['slope_2', 'restecg_1'],axis=1)
x_test=x_test.drop(['slope_2', 'restecg_1'],axis=1)
print(x_train.shape)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
model=RandomForestClassifier(n_estimators=200)
model.fit(x_train,y_train)
sc=model.score(x_test,y_test)
print(sc)
model=GaussianNB()
model.fit(x_train,y_train)
sc=model.score(x_test,y_test)
pred=model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(sc)
model=SVC()
model.fit(x_train,y_train)
sc=model.score(x_test,y_test)
pred=model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(sc)
