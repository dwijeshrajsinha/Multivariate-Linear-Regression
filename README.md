# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step1

Import the required libraries such as NumPy, Pandas, and Scikit-learn.

Step2

Load the dataset and separate the independent variables (features) and dependent variable (target).

Step3

Split the dataset into training and testing sets.

Step4

Create and train the Multivariate Linear Regression model using the training data.

Step5

Predict the output using the test data and display the regression coefficients, intercept, and predicted values.


## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model,metrics
from sklearn.datasets import fetch_california_housing

#load the boston dataset
boston=fetch_california_housing()

#defining feature matrix(X) and response vector(y)
X=boston.data
y=boston.target

#splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)

#create linear regression object
reg=linear_model.LinearRegression()

#train the model using the training sets
reg.fit(X_train,y_train)

#regression coefficients
print("Coefficients:",reg.coef_)

#variance score: 1 means perfect prediction
print("Variance score: {}".format(reg.score(X_test,y_test)))

#plot for residual error
##setting plot style
plt.style.use('fivethirtyeight')

##plotting residual errors in training data
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train,color="green",s=10,label="Train data")

##plotting residual errors in test data
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,color="blue",s=10,label="Test data")

##plotting line for zero residual error
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)

##plotting legend
plt.legend(loc="upper right")

##plot title
plt.title("Residual errors")

##method call for showing the plot
plt.show()
```
## Output:
<img width="1023" height="754" alt="image" src="https://github.com/user-attachments/assets/034e7b87-158c-48d2-a89f-5579f5c8e66b" />
## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
