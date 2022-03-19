import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
train_df=pd.read_csv('../datasets/train.csv')
test_df=pd.read_csv('../datasets/test.csv')
# print(train_df.head(10))
# print(train_df.shape)
# print(train_df.columns)
# print(train_df.describe(include='all'))
# total=train_df.isnull().sum().sort_values(ascending=False)
# percent_1=train_df.isnull().sum()/train_df.isnull().count()*100
# percent_2=(round(percent_1,1)).sort_values(ascending=False)
# missing_data=pd.concat([total,percent_2],axis=1,keys=['total','%'])
# print(missing_data)
# survived='survived'
# not_survived='not survived'
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
# women=train_df[train_df['Sex']=='female']
# men=train_df[train_df['Sex']=='male']
# ax=sns.distplot(women[women['Survived']==1].Age.dropna(),bins=18,label=survived,ax=axes[0],kde=False)
# ax=sns.distplot(women[women['Survived']==0].Age.dropna(),bins=40,label=not_survived,ax=axes[0],kde=False)
# ax.legend()
# ax.set_title('female')
# ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
# ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1],kde = False)
# ax.legend()
# ax.set_title('Male')
# Facetgrid=sns.FacetGrid(train_df,row='Embarked',size=4.5,aspect=1.6)
# Facetgrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
# Facetgrid.add_legend()
# sns.barplot(x='Pclass',y='Survived',data=train_df)
# Facetgrid=sns.FacetGrid(train_df,col='Survived',row='Pclass',size=4.5,aspect=1.6)
# Facetgrid.map(plt.hist,'Age', alpha=0.5,bins=18 )
# Facetgrid.add_legend()
plt.show()
data=[train_df,test_df]
for dataset in data:
    dataset['relatives']=dataset['SibSp']+dataset['Parch']
    dataset.loc[dataset['relatives'] > 0,'not_alone']=0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone']=dataset['not_alone'].astype(int)
# print(train_df['not_alone'].value_counts())
train_df = train_df.drop(['PassengerId'], axis=1)
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]
for dataset in data:
    mean=train_df['Age'].mean()
    std=test_df['Age'].std()
    isnull=dataset['Age'].isnull().sum()
    rand=np.random.randint(mean-std,mean+std,size=isnull)
    age_slice=dataset['Age'].copy()
    age_slice[np.isnan(age_slice)]=rand
    dataset['Age']=age_slice
    dataset['Age']=train_df['Age'].astype(int)
data=[train_df,test_df]
for dataset in data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare']=dataset['Fare'].astype(int)
titles={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', \
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] =dataset['Title'].map(titles)
    dataset['Title']=dataset['Title'].fillna(0)
train_df=train_df.drop(['Name','Ticket'],axis=1)
test_df=test_df.drop(['Name','Ticket'],axis=1)
le=LabelEncoder()
train_df['Sex']=le.fit_transform(train_df['Sex'])
test_df['Sex']=le.fit_transform(test_df['Sex'])
train_df['Embarked']=le.fit_transform(train_df['Embarked'])
data=[train_df,test_df]
for dataset in data:
    dataset['Age']=dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[dataset['Age'] > 66, 'Age'] = 6
data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]
for dataset in data:
    dataset['Age-Class']=dataset['Age']*dataset['Pclass']
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# print(train_df['Age'].value_counts())
# print(train_df.info())
# print(train_df.isnull().sum())
# print(train_df.columns)
# print(train_df.head(10))
print(train_df.columns)
x_train1=train_df[[ 'Pclass', 'Sex', 'Age', 'SibSp',  'Fare',
       'Embarked', 'relatives',  'Deck', 'Title', 'Age-Class',
       'Fare_Per_Person'
        ]]
y_train1=train_df['Survived']
model1 =RandomForestClassifier(n_estimators=100)
rfe=RFE(model1)
fit = rfe.fit(x_train1,y_train1)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train1,y_train1,test_size=0.20,random_state=1)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score
model=GaussianNB()
model.fit(x_train,y_train)
scorev=model.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
print("Decision tree"+str(clf.score(x_test,y_test)))
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
print("Random tree"+str(random_forest.score(x_test,y_test)))
# score1=precision_score(y_test,pred)
# score2=recall_score(y_test,pred)
print("Gaussian nb"+str(scorev))
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(x_train,y_train)
score2=model1.score(x_test,y_test)
print("Logistic regression"+str(score2))