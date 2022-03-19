import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
data=pd.read_csv('../datasets/insurance.csv')
# print(data.head())
# print(data.describe())
le =LabelEncoder()
data['sex']=le.fit_transform(data['sex'])
data['smoker']=le.fit_transform(data['smoker'])
data['region']=le.fit_transform(data['region'])
# sns.jointplot(x=data['age'],y=data['expenses'])
# sns.jointplot(x=data['children'],y=data['expenses'])
# sns.jointplot(x=data['smoker'],y=data['expenses'])
# plt.show()
x=data.drop(['expenses'],axis=1).values
# x=data.drop(['expenses'],axis=1)
y=data['expenses'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model = LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(math.sqrt(mean_squared_error(pred,y_test)))
plt.scatter(y_test,pred,color='yellow')
plt.plot(x_train,model.predict(x_train),color='black')
plt.show()
print(model.coef_)
print(model.intercept_)


