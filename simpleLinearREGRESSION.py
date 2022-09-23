import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/Admin/Desktop/PYTHON/PROJECT/Salary_Data.csv")
print(data.shape)
print(data.isna().sum())
print(data.columns)
X=data['YearsExperience'].values
Y=data['Salary'].values

print(X.shape)
print(Y.shape)

X=X.reshape(X.shape[0],1)
Y=Y.reshape(X.shape[0],1)
print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
Y_custom=reg.predict([[2.7]])

import matplotlib.pyplot as plt
plt.scatter(X_test,Y_test,c='b',label="Orignal data")
plt.plot(X_test,Y_pred,c='r',label="Predicted data")
plt.legend()
plt.show()
