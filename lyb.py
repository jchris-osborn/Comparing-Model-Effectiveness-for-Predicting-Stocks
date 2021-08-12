import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from numpy import inf


file1 = pd.read_csv('C:/Users/osbo3/Dropbox/Financial Enggineering/Project/Data/lyb.us.txt', sep=",")

#creating DataFrame
data = pd.DataFrame(file1)
print(data.head())

#At the volume closing price relationship
data.plot(x='Volume', y='Close')
plt.title("Close v Volume")
plt.show()

#veiwing the outlier in volume before the log transofrmation
data['Volume'].plot().get_figure()
sns.boxplot(x=data['Volume'])
plt.show()

data.hist(bins=10)
plt.show()

#veiwing the trend in volume befor transformation
sns.lmplot(x='Close', y='Volume', data=data)
plt.show()

#log transformation
data['Volume'] = np.log(data['Volume'])
data[data['Volume'] == -inf] = 0
data = data[(data.T != 0).any()]
data['Volume'].plot().get_figure()
plt.show()

#veiwing the outlier in volume after the log transofrmation
sns.boxplot(x=data['Volume'])
plt.show()

data.hist(bins=10)
plt.show()

#veiwing the trend in volume after transformation
sns.lmplot(x='Close', y='Volume', data=data)
plt.show()

#visualizing the time series data of the stock
data.plot(x='Date', y='Close')
plt.show()

#visualizing the missing in the data
msno.bar(data)
plt.show()

#Splitting the data into train and test
msk = np.random.rand(len(data)) < 0.7

train = data[msk]
test = data[~msk]

#Modeling the data with SVR LSVR kNN Random Forrest and Bagging
#Training the training data and predicting the test data and computing the RSME

x_train = data.columns.tolist()
x_train = [c for c in x_train if c not in ['Date', 'OpenInt', 'Close']]
y_train = "Close"

#SVR
svr = SVR()
svr.fit(train[x_train], train[y_train])
predictions1 = svr.predict(test[x_train])

print('SVR')
print(mean_squared_error(predictions1, test[y_train]))
print(r2_score(predictions1, test[y_train]))
print(mean_absolute_error(predictions1, test[y_train]))

#LSVR
lsvr = svm.LinearSVR(max_iter=10000)
lsvr.fit(train[x_train], train[y_train])
predictions2 = lsvr.predict(test[x_train])

print('LSVR')
print(mean_squared_error(predictions2, test[y_train]))
print(r2_score(predictions2, test[y_train]))
print(mean_absolute_error(predictions2, test[y_train]))

#kNN
neigh = KNeighborsRegressor(n_neighbors=4)
neigh.fit(train[x_train], train[y_train])
predictions3 = neigh.predict(test[x_train])

print('kNN')
print(mean_squared_error(predictions3, test[y_train]))
print(r2_score(predictions3, test[y_train]))
print(mean_absolute_error(predictions3, test[y_train]))

#Random Forrest
regr = RandomForestRegressor()
regr.fit(train[x_train], train[y_train])
predictions4 = regr.predict(test[x_train])

print('Random Forest')
print(mean_squared_error(predictions4, test[y_train]))
print(r2_score(predictions4, test[y_train]))
print(mean_absolute_error(predictions4, test[y_train]))

#Bagging
bag = BaggingRegressor()
bag.fit(train[x_train], train[y_train])
predictions5 = bag.predict(test[x_train])

print('Bagging')
print(mean_squared_error(predictions5, test[y_train]))
print(r2_score(predictions5, test[y_train]))
print(mean_absolute_error(predictions5, test[y_train]))