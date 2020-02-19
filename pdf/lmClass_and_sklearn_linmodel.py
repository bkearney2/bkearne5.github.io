#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:01:33 2019

@author: bkearney

Main linear regression code. Shuffles and separates diabetes BMI dataset
    into training and testing sets (lines161-193).
Uses class lmClass (lines 195-208) and sklearn (lines 210-233)
Plots testing x vs testing y and testing x vs predicted y for both inclass code and sklearn

Talk about pearson correlation / f test, what trend is it (positive/negative) ?
Writeup: Simple linear regression predicted response variable "target" based on
     predictor variable "BMI" in diabetes data set. From the F-test p-value and
     correlation coefficients seen in the output, we can conclude that there is
     some linear relationship between the variables, but generally most of the variation
     cannot be attributed to the variation between target and BMI variables (R^2 < 0.5).
     (See output for F testing, R^2, and plotting)
"""

import os, sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import datasets
import scipy.stats as stats
import math

# --------------------------------------------------------------------------
# Assignment: Revise the linear regression code we went through in class by 
# replacing the current function with a class.
# --------------------------------------------------------------------------

class lmClass:
    def _init_(self):
        pass
        
    def fit(self,x,y):
        n = len(x)
        
        x_bar = np.mean(x)
        y_bar = np.mean(y)

        S_yx = np.sum((y - y_bar) * (x - x_bar))
        temp = (y-y_bar) * (x - x_bar)
        #print(np.shape(temp), "SHAPE")
        S_xx = np.sum((x - x_bar)**2)

        # ====== estimate beta_0 and beta_1 ======
        beta_1_hat = S_yx / S_xx # also equal to (np.cov(x, y))[0, 1] / np.var(x)
        beta_0_hat = y_bar - beta_1_hat * x_bar

        # ====== estimate sigma ======
        # residual
        y_hat = beta_0_hat + beta_1_hat * x
        r = y - y_hat
        sigma_hat = np.sqrt(sum(r**2) / (n-2))

        # ====== estimate sum of squares ======
        # total sum of squares
        SS_total = np.sum((y - y_bar)**2)
        # regression sum of squares
        SS_reg = np.sum((y_hat - y_bar)**2)
        # residual sum of squares
        SS_err = np.sum((y - y_hat)**2)

        # ====== estimate R2: coefficient of determination ======
        R2 = SS_reg / SS_total

        # ====== R2 = correlation_coefficient**2 ======
        correlation_coefficient = np.corrcoef(x, y)
        delta = correlation_coefficient[0, 1]**2 - R2

        # ====== estimate MS ======
        # sample variance
        MS_total = SS_total / (n-1)
        MS_reg = SS_reg / 1.0
        MS_err = SS_err / (n-2)
    
        # ====== estimate F statistic ======
        F = MS_reg / MS_err
        F_test_p_value = 1 - stats.f._cdf(F, dfn=1, dfd=n-2)
    
        # ====== beta_1_hat statistic ======
        beta_1_hat_var = sigma_hat**2 / ((n-1) * np.var(x))
        beta_1_hat_sd = np.sqrt(beta_1_hat_var)
    
        # confidence interval
        z = stats.t.ppf(q=0.025, df=n-2)
        beta_1_hat_CI_lower_bound = beta_1_hat - z * beta_1_hat_sd
        beta_1_hat_CI_upper_bound = beta_1_hat + z * beta_1_hat_sd
    
        # hypothesis tests for beta_1_hat
        # H0: beta_1 = 0
        # H1: beta_1 != 0
        beta_1_hat_t_statistic = beta_1_hat / beta_1_hat_sd
        beta_1_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_1_hat_t_statistic), df=n-2))
    
        # ====== beta_0_hat statistic ======
        beta_0_hat_var = beta_1_hat_var * np.sum(x**2) / n
        beta_0_hat_sd = np.sqrt(beta_0_hat_var)
    
        # confidence interval
        beta_0_hat_CI_lower_bound = beta_0_hat - z * beta_0_hat_sd
        beta_1_hat_CI_upper_bound = beta_0_hat + z * beta_0_hat_sd
        beta_0_hat_t_statistic = beta_0_hat / beta_0_hat_sd
        beta_0_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_0_hat_t_statistic), df=n-2))
    
        # confidence interval for the regression line
        sigma_i = 1.0/n * (1 + ((x - x_bar) / np.std(x))**2)
        y_hat_sd = sigma_hat * sigma_i
    
        y_hat_CI_lower_bound = y_hat - z * y_hat_sd
        y_hat_CI_upper_bound = y_hat + z * y_hat_sd

    
        self.beta_1_hat = beta_1_hat
        self.beta_0_hat = beta_0_hat
        self.sigma_hat = sigma_hat
        self.y_hat = y_hat
        self.R2 = R2
        self.F_statistic = F
        self.F_test_p_value = F_test_p_value
        self.MS_error = MS_err
        self.beta_1_hat_CI = np.array([beta_1_hat_CI_lower_bound, beta_1_hat_CI_upper_bound])
        self.beta_1_hat_standard_error = beta_1_hat_sd
        self.beta_1_hat_t_statistic = beta_1_hat_t_statistic
        self.beta_1_hat_t_test_p_value = beta_1_hat_t_test_p_value
        self.beta_0_hat_standard_error = beta_0_hat_sd
        self.beta_0_hat_t_statistic = beta_0_hat_t_statistic
        self.beta_0_hat_t_test_p_value = beta_0_hat_t_test_p_value
        self.y_hat_CI_lower_bound = y_hat_CI_lower_bound
        self.y_hat_CI_upper_bound = y_hat_CI_upper_bound

        return self
# --------------------------------------------------------------------------
# set up plotting parameters
# --------------------------------------------------------------------------
line_width_1 = 2
line_width_2 = 2
marker_1 = '.' # point
marker_2 = 'o' # circle
marker_size = 12
line_style_1 = ':' # dotted line
line_style_2 = '-' # solid line

#---------------------------
# plotting function
#---------------------------
def plot_func(x, y, y_hat, title, dot_color, line_color):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, color=dot_color, label='testing x vs testing y',marker=marker_1, linewidth=line_width_1)
    ax.scatter(x, y_hat, color = 'black',label = 'testing x vs predicted y', marker=marker_2, linewidth= line_width_1)
    ax.plot(x, y_hat, color=line_color, label='predicted', linewidth=line_width_1)
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    n = len(x)
    ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
    ax.plot([x_bar, x_bar], [np.min(y), np.max(y)],color='black', linestyle=':', linewidth=line_width_1)
    ax.set_xlabel('BMI')
    ax.set_ylabel('Target')
    ax.set_title("Linear regression"+title)
    ax.legend(loc='lower right', fontsize=9)
    fig.show()
    
# ----------------------------------------------------
# data input
# ----------------------------------------------------
# Extract diabetes bunch dataset
diabetes = datasets.load_diabetes()
#print(diabetes.feature_names)
# Select x and y variables from diabetes
bmi_x = diabetes.data[:,2]
bmi_y = diabetes.target

# Create 2d array of x and y variables to keep them attached during random selection
bmi_2d = np.vstack((bmi_x,bmi_y))
#print("Dimensions (rows/cols): ", np.shape(bmi_2d))

# Shuffle columns 2d array to simulate random selection
bmi_2d_shuffled = np.random.permutation(bmi_2d.T).T
#print("dimensions (rows/cols) shuffled: ", np.shape(bmi_2d_shuffled))
#print(bmi_2d_shuffled)
#print("First value unshuffled: ", bmi_2d[:,0])
#print("First value shuffled: ", bmi_2d_shuffled[:,0])

# Make new shuffled x and y arrays
bmi_x_shuffled = bmi_2d_shuffled[0,:]
bmi_y_shuffled = bmi_2d_shuffled[1,:]
#print(bmi_y_shuffled)

# Select 20 samples to testing set and rest to training
bmi_x_train = bmi_x_shuffled[:-20]
bmi_x_test = bmi_x_shuffled[-20:]
bmi_y_train = bmi_y_shuffled[:-20]
bmi_y_test = bmi_y_shuffled[-20:]
len_x_train = len(bmi_x_train)
len_x_test = len(bmi_x_test)

# --------------------------------------------------------------------------
# linear regression
# --------------------------------------------------------------------------
n = len(bmi_x)

# do linear regression using my class
lm1 = lmClass().fit(bmi_x_train,bmi_y_train)
#plot_func(bmi_x_train, bmi_y_train, lm1.y_hat, " -- training set data", 'red', 'blue')

print ("Beta 1 hat (manual code): ", lm1.beta_1_hat)
print ("Beta 0 hat (manual code): ", lm1.beta_0_hat)

# Manually calculate yhat of testing data
bmi_yhat_test = lm1.beta_0_hat + lm1.beta_1_hat * bmi_x_test
plot_func(bmi_x_test,bmi_y_test, bmi_yhat_test, " -- BMI vs target, class code",'red', 'blue')

# --------------------------------------------------------------------------
# linear regression using sklearn
# --------------------------------------------------------------------------

# Create linear regression object
reg1 = linear_model.LinearRegression()

# Train the model using the training sets
x_reshaped = bmi_x_train.reshape((len_x_train, 1))
x_test_reshaped = bmi_x_test.reshape((len_x_test, 1))
reg1.fit(x_reshaped, bmi_y_train)

# Get parameters of regression
diabetes_params = reg1.get_params()
#print(diabetes_params)
print("Beta 1 hat (sklearn): ",reg1.coef_[0])
print("Beta 0 hat (sklearn): ",reg1.intercept_)

# Make predictions using the testing set
bmi_y_test_pred = reg1.predict(x_test_reshaped)
#print(diabetes_y_test_pred)
y_reshaped = bmi_y_train.reshape((len_x_train, 1))

# plot testing x vs testing y and predicted y 
plot_func(bmi_x_test, bmi_y_test, bmi_y_test_pred, " -- BMI vs target, sklearn", 'green', 'purple')
print ("F statistic: ", lm1.F_statistic)
print ("F-test p-value: ",lm1.F_test_p_value)
r2_sqrt = math.sqrt(lm1.R2)
print ("Coefficient of Determination (r^2), Correlation Coefficient (r): ", lm1.R2, r2_sqrt)
