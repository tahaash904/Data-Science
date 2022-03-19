import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error
train=pd.read_csv('../datasets/walmart/train.csv')
test=pd.read_csv('../datasets/walmart/test.csv')
stores=pd.read_csv('../datasets/walmart/stores.csv')
features=pd.read_csv('../datasets/walmart/features.csv')
# for i in [train,test,stores,features]:
#     print(i.head(10))
data=train.merge(features,on=['Store','Date'],how='inner').merge(stores,on=['Store'],how='inner')
data=data.drop(['IsHoliday_x'],axis=1)
data=data.sample(frac=0.8)
# print(data.columns)
data=data.fillna(0)
# total=data.isnull().sum().sort_values(ascending=False)
# percent1=data.isnull().sum()/data.isnull().count()*100
# percent2=(round(percent1,1)).sort_values(ascending=False)
# missing=pd.concat([total,percent2],axis=1,keys=['Total','%'])
# print(missing)
data=data.sort_values(by='Date')
sizes=stores['Type'].value_counts()
print(sizes)
# f,ax=plt.subplots(figsize=(8,6))
# fig=sns.boxplot(x='Type',y='Size',data=stores)
# weekly_sales1=pd.concat([stores['Type'],data['Weekly_Sales']],axis=1)
# f,ax=plt.subplots(figsize=(8,6))
# fig=sns.boxplot(x='Type',y='Weekly_Sales',data=weekly_sales1,showfliers=False)
# f,ax=plt.subplots(figsize=(8,6))
# fig=sns.boxplot(x='IsHoliday_y',y='Weekly_Sales',data=data,showfliers=False)
# corr=data.corr()
# plt.figure(figsize=(15,10))
# sns.heatmap(corr,annot=True)
# plt.show()
data['Weekly_Sales']=abs(data['Weekly_Sales'])
data['Weekly_Sales']=data['Weekly_Sales'].fillna(mean(data['Weekly_Sales']))
# print(data['Weekly_Sales'].describe())
# print(data['Weekly_Sales'].isnull().sum())
data['Year']=pd.to_datetime(data['Date']).dt.year
data['Day']=pd.to_datetime(data['Date'],format="%Y-%m-%d").dt.day
data["Days to Next Christmas"] = (pd.to_datetime(data["Year"].astype(str)+"-12-31", format="%Y-%m-%d") -
                                   pd.to_datetime(data["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data=pd.concat([data,pd.get_dummies(data['Type'],prefix='Type')],axis=1)
data=data.drop(['Type','Date','MarkDown1', 'MarkDown2',  'MarkDown4', 'MarkDown5','MarkDown3'],axis=1)
# print(data.columns)
# print(data.head(80))
x=data.drop(['Weekly_Sales'],axis=1)
y=data['Weekly_Sales']
print(x.shape)
print(y.shape)
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
model1 = DecisionTreeRegressor(random_state=0)
rfe=RFE(model1,3)
fit = rfe.fit(x,y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
print(x.columns)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import MinMaxScaler
xc=MinMaxScaler()
x_train=xc.fit_transform(x_train)
x_test=xc.fit_transform(x_test)
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
# model=KNeighborsRegressor(n_neighbors=15,n_jobs=4)
# model.fit(x_train,y_train)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x_train,y_train)
test_score=dt.score(x_test,y_test)*100
train_score=dt.score(x_train,y_train)*100
print(train_score)
print(test_score)
pred=dt.predict(x_test)
print(np.sqrt(mean_squared_error(pred,y_test)))
plt.scatter(dt.predict(x_test),y_test)
plt.show()
