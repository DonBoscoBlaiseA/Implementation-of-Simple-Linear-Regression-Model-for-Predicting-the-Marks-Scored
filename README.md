# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary packages and libraries.
2. Access the csv file and segregate the data to variables.
3. Split training and test data and import linear regression model.
4. Train and test the values and plot it using matplotlib.
5. Finally get the mse,mae and rmse values.

## Program:
```
/*
Program to implement simple linear regression model for predicting the marks scored.
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/86cba780-9e56-445e-a6a1-2aba780a433b.png" width="200">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/e93e90d2-b242-47cc-b232-8bf85c688a29.png" width="200">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/70f32fef-5fb8-43a8-b5b5-f08ef5e86665.png" width="200">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/4f3f0684-1a85-47de-8b1f-3a6440690157.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/69957f4a-47e4-4f4e-84ee-7bcfbd85c438.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/e13dbd8c-8d65-46ee-ad51-7448d8ab6caf.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/f9b4ed41-228b-4279-a093-e01b94dd7bc8.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/c60c5b8c-dde4-4145-b056-6e7a7697e79e.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/d267bc2a-8d59-469b-99b2-8e22ad46d2cf.png" width="600">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/140850829/1d81dc59-31f2-4202-9a01-28ef9cd364f5.png" width="400">  

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
