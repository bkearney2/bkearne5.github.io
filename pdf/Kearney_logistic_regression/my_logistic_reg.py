#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:20:26 2019

Gradient descent-based logistic regression class. 

Learning rate and number of iterations can be modified,
stopping criterion also applies when delta_j < 0.00001

See attached .pdf file for iterations over cost plot,
iris dataset predictions, and sklearn application. 

@author: bkearney
"""

import numpy as np

#Logistic regression class

class LogReg:
    def __init__(self, learn_rate = .05, iterations = 100000, fit_intercept=True):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        
    # Add column of 1s to x array before features
    def __add_intercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis = 1)
        
    # sigmoid function
    def __g_sigmoid(self, z): 
        g_z = 1 / (1 + np.exp(-z))
        return g_z
            
    # cost function for j(theta), len(y) = m
    def __get_cost(self, h, y):
        cost = -(1/len(y)) * np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        return cost
    
    def __get_gradient(self, x, y, h):
        gradient = np.dot(x.T, h-y) / y.size
        return gradient
    
    def do_gradient_descent(self, x, y):
        if self.fit_intercept:
            x = self.__add_intercept(x)
            
        # Create arrays that record cost and #iterations over iterations 
        cost_history = np.array([])
        iterations_history = np.array([])
        
        # Initialize delta J
        delta_j = 1
        
        for i in range(self.iterations):
            if (i == 0):
                self.theta = np.zeros(x.shape[1])
                z = np.dot(x, self.theta)
                h = self.__g_sigmoid(z)
                c = self.__get_cost(h,y)   # Initial cost
            
            else: 
                gradient = self.__get_gradient(x, y, h)
                self.theta -= self.learn_rate * gradient
                z = np.dot(x, self.theta)
                h = self.__g_sigmoid(z)
                new_c = self.__get_cost(h,y)
                delta_j = c - new_c # only record delta_j when iterations>0
                c = new_c
                
            iterations_history = np.append(iterations_history, i)
            cost_history = np.append(cost_history, c)
            
            # Stopping criterion for gradient descent
            if (delta_j < 0.00001): 
                coefs = self.theta[1:5]
                intercept = self.theta[0]
                break

        self.iterations_history = iterations_history
        self.cost_history = cost_history
        self.c = c
        self.coefs = coefs
        self.intercept = intercept

    def predict(self, x, threshold):
        if self.fit_intercept:
            x = self.__add_intercept(x)
        prob = self.__g_sigmoid(np.dot(x, self.theta))
        return prob >= threshold

