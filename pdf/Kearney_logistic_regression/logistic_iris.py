#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:23:56 2019

Application of iris dataset to logistic regression class (my_logistic_reg)

@author: bkearney
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import random
from sklearn.linear_model import LogisticRegression
from my_logistic_reg import LogReg


iris = datasets.load_iris()
X = iris.data[50:150,]
y = iris.target[50:150] - 1
rand_sample = random.randint(0,99) # Randomly select out of 100

# Choose only one sample for testing using random
testing_X = X[rand_sample:rand_sample+1,] 
testing_y = y[rand_sample]

# Other 99 samples go to training
training_X = np.delete(X, rand_sample, axis=0)
training_y = np.delete(y, rand_sample)

# Use LogReg class for logistic regression
model = LogReg()
model.do_gradient_descent(X, y)

prediction = model.predict(testing_X, 0.5) # 0.5 is default threshold, can be modified

# decide between two flowers with prediction
if (prediction.mean() == 0):
    flower_predict = iris.target_names[1]
else: flower_predict = iris.target_names[2]

######## Plotting #########
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(model.iterations_history, model.cost_history)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost - J')
ax.set_title("Logistic Regression: Cost vs Iterations, Learning rate = "+ str(model.learn_rate))

######## skLearn logistic regression #########
clf = LogisticRegression(random_state=0, solver='liblinear',multi_class='auto').fit(training_X, training_y)

# mini combined cost function used to test sklearn's intercept and coefs
def compute_cost_with_coefs(X, y, theta, fit_intercept = True):
     if fit_intercept:
         intercept = np.ones((X.shape[0], 1))
         X = np.concatenate((intercept, X), axis = 1)
         
     z = np.dot(X, theta)
     h = 1 / (1 + np.exp(-z))
     c = -(1/len(y)) * np.sum(y*np.log(h)+(1-y)*np.log(1-h))
     return c

theta_combined = np.append(clf.intercept_,clf.coef_)
sklearn_cost = compute_cost_with_coefs(training_X, training_y, theta_combined)

######## Output section ########
print('\n')
print("Testing flower number: ", rand_sample+51)
print("Target = ", testing_y, "Prediction = ", flower_predict)

print("My final cost: ", model.c)
print("My coefficients: ",model.theta[1:5],"My intercept: ",model.theta[0] )

print('\n')
print("sklearn coefficients: ",clf.coef_, "Intercept: ",clf.intercept_)
print("sklearn cost = ",sklearn_cost)
