#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 1 20:14:57 2019

@author: bkearney, rosh, brandon, aaron, connor
"""

import matplotlib.pyplot as plt
import numpy as np

def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ

def Relu(Z):
    return np.maximum(0, Z)

def dRelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
    
class dlnet: 
    def __init__(self, x, y):
        self.X = x 
        self.Y = y
        self.Yh = np.zeros((1, self.Y.shape[1]))
        self.param = {} #weights and biases
        self.ch = {} #cache variable
        self.loss = []
        self.sam = self.Y.shape[1]
        self.threshold = 0.5
        
        self.dims = [10, 10, 10, 1]
        self.lr = 0.01

    def nInit(self):
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        self.param['W3'] = np.random.randn(self.dims[3], self.dims[2]) / np.sqrt(self.dims[2])
        self.param['b3'] = np.zeros((self.dims[3], 1))

        return
    
    def forward(self):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = Relu(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1
        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Relu(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2
        
        Z3 = self.param['W3'].dot(A2) + self.param['b3']
        A3 = Sigmoid(Z3)
        self.ch['Z3'], self.ch['A3'] = Z3, A3
        self.Yh = A3
        loss = self.nloss(A3)
        return self.Yh, loss
    
    def nloss(self, Yh):
        loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
        return loss
    
    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh) - np.divide(1 - self.Y, 1 - self.Yh))
        dLoss_Z3 = dLoss_Yh * dSigmoid(self.ch['Z3'])
        dLoss_A2 = np.dot(self.param["W3"].T, dLoss_Z3)
        dLoss_W3 = 1. / self.ch['A2'].shape[1] * np.dot(dLoss_Z3, self.ch['A2'].T)
        dLoss_b3 = 1. / self.ch['A2'].shape[1] * np.dot(dLoss_Z3, np.ones([dLoss_Z3.shape[1], 1]))
        dLoss_Z2 = dLoss_A2 * dSigmoid(self.ch['Z2'])
        dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
        dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
        dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))
        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
        dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
        dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        self.param["W3"] = self.param["W3"] - self.lr * dLoss_W3
        self.param["b3"] = self.param["b3"] - self.lr * dLoss_b3
        return
    
    def pred(self, x, y):
        self.X = x
        self.Y = y
        comp = np.zeros((1, x.shape[1]))
        pred, loss = self.forward()
        for i in range(0, pred.shape[1]):
            if pred[0, i] > self.threshold:
                comp[0, i] = 1
            else:
                comp[0, i] = 0
                
        return pred, comp
    
    def gd(self, X, Y, iter=15000):
        np.random.seed(1)
        self.nInit()
        
        for i in range(0, iter):
            Yh, loss = self.forward()
            self.backward()
                
            if i % 1000 == 0:
#                print("Cost after iteration %i: %f" % (i, loss))
                self.loss.append(loss)
                
        plt.plot(np.squeeze(self.loss))
        plt.ylabel('Loss')
        plt.xlabel('Iterations (thousands)')
        plt.title("Loss function, Lr =" + str(self.lr)+ ", nn dimensions = "+ str(self.dims))
        plt.show()
        
        return loss