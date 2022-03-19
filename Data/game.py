import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
game = pd.read_csv('../datasets/games.csv')
print(game.columns)
print(game.shape)
# plt.hist(game['average_rating'])
# plt.show()
# print(game[game['average_rating']==0].iloc[0])
# print(game[game['average_rating']>0].iloc[0])
game = game[game['users_rated'] > 0]
game = game.dropna(axis=0)
# plt.hist(game['average_rating'])
# cormat = game.corr()
# fig = plt.figure(figsize=(12,9))
# sns.heatmap(cormat, vmax=.8, square=True)
# plt.show()
coloumns = game.columns.tolist()
coloumns = [c for c in coloumns if c  not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]
target = 'average_rating'
train = game.sample(frac=.8,random_state=1)
test = game.loc[~game.index.isin(train.index)]
print(train.shape)
print(test.shape)
from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(train[coloumns],train[target])
from sklearn.metrics import mean_squared_error
predictions = clf.predict(test[coloumns])
print(mean_squared_error(predictions, test[target]))
y=game['average_rating']
x=game.drop(['average_rating','type','name','id'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
clf1 = LinearRegression()
clf1.fit(X_train,y_train)
predictions = clf1.predict(X_test)
print(mean_squared_error(predictions, y_test))