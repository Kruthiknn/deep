from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
data=fetch_california_housing()
x=data.data
y=data.target
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
# score=accuracy_score(y_pred=y_pred,y_true=Y_test)


print(model.score(X_test,Y_test))
plt.scatter(x=y_test,y=y_pred)
plt.show()

import keras
from keras.models import Sequential
from keras.layers import dense
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
#from keras.utils import to_california
model=Sequential()
#load the dataset
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
print(X_train.shape,X_test.shape)
