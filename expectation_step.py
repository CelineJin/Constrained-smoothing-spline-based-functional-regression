#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 07:31:12 2023

@author: celinejin
"""
from scipy.optimize import minimize
import numpy as np

class expectation_step():
    def __init__(self, t, b_est, M, c, Phi, y, initial_guess, bounds):
        self.x = None
        self.t = t
        self.b_est = b_est
        self.M = M
        self.c = c
        self.Phi = Phi
        self.y = y
        self.initial_guess = initial_guess
        self._bounds = bounds
        self.obj_func = None
        self.result = None
        self.objective()

    def objective(self): # M is the X matrix out of the maximization step
        X = self.M
        nsample = X.shape[0]
        Kb = len(self.x)
        obj = self.b_est[0] #alpha
        obj += self.b_est[:,1]*X[:,1].reshape(-1,1)
        obj += self.b_est[:,2]*X[:,2].reshape(-1,1)
        Mk = np.zeros((nsample,Kb))
        for k in range(Kb):  
            for i in range(nsample):
                fac_Mk = np.matmul(self.Phi.transpose(), np.maximum(self.t-self.x[k],0)).reshape(-1,1)
                Mk[i,k] = np.matmul(self.c[i,:],fac_Mk)
            obj += self.b_est[:,k+2]*Mk[:,k].reshape(-1,1)
        obj_func = sum((self.y-obj)**2)
        self.obj_func = obj_func
        return obj_func
    
    def optimization(self):
        func = lambda x: self.objective()
        result = minimize(func, x0=self.initial_guess, args = (self.t,self.b_est,self.X,self.c,self.Phi,self.y), bounds = self.bounds)
        self.result = result
        return result
