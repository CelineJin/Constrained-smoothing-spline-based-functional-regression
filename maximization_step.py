#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:57 2023
    Use:
        Given the knots, estimate the coefficients b_i, i=0,...,K_b+1
    Inputs:
        knots
        x: the sample functions
        V: eigenfunctions of cov of x; each column is one eigenfunction (output of KLExpansion)
        c: entry cij is the projection of ith. x observation on jth. eigenfunction (output of KLExpansion)
        ytrue: scalar response
    Outputs:
        coef: estimates of b
        ypred: predicted scalar response
@author: celinejin
"""

import numpy as np
from sklearn.linear_model import LinearRegression
class maximization_step():
    def __init__(self, knots, x, V, c, ytrue):
        self.knots = knots
        self.x = x
        self.V = V
        self.c = c
        self.ytrue = ytrue
        self.X = None
        self.ypred = None
        self.b_estimate = None
        
        
    def linFeatureConstruction(self):
        t = np.linspace(0,1,num=self.x.shape[1])
        Kx = self.c.shape[1]
        # onevec = np.ones((self.x.shape[0],1))
        
        delt = np.ones((self.V.shape[0],1)).flatten()
        # delt = t[1]*np.ones((self.V.shape[0],1)).flatten()
        M0 = np.matmul(self.V.transpose(),delt).reshape(-1,1) # a column vector, card = Kx
        x0 = np.matmul(self.c,M0).reshape(-1,1)
        
        # X = np.concatenate((onevec, x0), axis=1)
        X = x0
        
        M1 = np.zeros((Kx,1)) # a column vector, card = Kx
        for ig in range(Kx):
            t = np.multiply(t, delt)
            M1[ig] = np.inner(t,self.V[:,ig])
        x1 = np.matmul(self.c,M1)
        
        X = np.concatenate((X, x1), axis=1) 
        
        for nknot in range(len(self.knots)):
            M = np.zeros((Kx,1))
            for ig in range(Kx):
                posfunc = np.where(t-self.knots[nknot]>0,t-self.knots[nknot],0)
                posfunc = np.multiply(posfunc,delt)
                M[ig] = np.inner(posfunc, self.V[:,ig])
            x = np.matmul(self.c,M)
            X = np.concatenate((X, x),axis=1)
        
        self.X = X
        return X
    
    def nonlinFeatureConstruction(self):
        t = np.linspace(0,1,num=self.x.shape[1])
        kx = self.c.shape[1]
        onevec = np.ones((self.x.shape[0],1))
        X = onevec
        
        delt = np.ones((self.V.shape[0],1)).flatten()
        M0 = np.matmul(self.V.transpose(),delt).reshape(-1,1)
        x0 = np.matmul(self.c,M0).reshape(-1,1)
        X = np.concatenate((X, x0), axis=1)
        
        M1 = np.zeros((kx,1))
        for ikx in range(kx):
            t = np.multiply(t,delt)
            M1[ikx] = np.inner(t,self.V[:,ikx])
        x1 = np.matmul(self.c, M1)
        X = np.concatenate((X, x1),axis=1)
        
        M2 = np.zeros((kx,1))
        for ikx in range(kx):
            t = np.multiply(t**2,delt)
            M2[ikx] = np.inner(t, self.V[:,ikx])
        x2 = np.matmul(self.c, M2)
        X = np.concatenate((X,x2), axis=1)
        
        M3 = np.zeros((kx,1))
        for ikx in range(kx):
            t = np.multiply(t**3,delt)
            M3[ikx] = np.inner(t,self.V[:,ikx])
        x3 = np.matmul(self.c, M3)
        X = np.concatenate((X,x3),axis=1)
        
        for nknot in range(len(self.knots)):
            M = np.zeros((kx,1))
            for ig in range(kx):
                posfunc = np.where(t-self.knots[nknot]>0,(t-self.knots[nknot])**3,0)
                posfunc = np.multiply(posfunc,delt)
                M[ig] = np.inner(posfunc, self.V[:,ig])
            x = np.matmul(self.c,M)
            X = np.concatenate((X, x),axis=1)
        
        self.X = X
        return X
        

    def maximization(self):
        lin_model = LinearRegression().fit(self.X,self.ytrue)
        ypred = lin_model.predict(self.X)
        self.ypred = ypred
        self.b_estimate = lin_model.coef_
        return lin_model.coef_, ypred
    
