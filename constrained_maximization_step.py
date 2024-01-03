#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:29:31 2023

@author: celinejin
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
from maximization_step import maximization_step
import os
import pandas as pd

class constrained_maximization_step():
    def __init__(self, knots, x, V, ytrue):
        self.knots = knots
        self.x = x
        self.V = V
        self.c = None
        self.ytrue = ytrue
        self.X = None
        self.b_estimate = None
        self.get_subfeatures()
        self.subfeaturesall = None
    
    
    def get_subfeatures(self):
        subfeaturesall = []
        T = self.x.shape[1] 
        dir_list = sorted(os.listdir("data"))
        dir_list = dir_list[1:]+[dir_list[0]]
        for ifile in range(len(dir_list)):
            Vsub = pd.read_csv('data/'+dir_list[ifile], header=None).values
            nT = Vsub.shape[1]
            xsub = self.x[:,:nT]
            subknots = np.where(self.knots<nT/T, self.knots, None).tolist()
            subknots = list(filter(lambda x: x is not None, subknots))
            subknots = np.array(subknots)
            # c = self.calprojection(self.xsub,self.Vsub)
            K_x = Vsub.shape[1]
            c = np.zeros((xsub.shape[0],K_x))
            for i in range(xsub.shape[0]):
                for k in range(K_x):
                    c[i,k] =  xsub[i,:]@Vsub[:,k]
            subfeatures = maximization_step(subknots, xsub, Vsub, c, self.ytrue)
            subfeature_matrix = subfeatures.nonlinFeatureConstruction()
            subfeaturesall.append(subfeature_matrix)
        self.subfeaturesall = subfeaturesall
        return subfeaturesall
        
        
    def constrained_least_square(self):
        
        K_x = self.V.shape[1]
        c = np.zeros((self.x.shape[0],K_x))
        for i in range(self.x.shape[0]):
            for k in range(K_x):
                c[i,k] =  self.x[i,:]@self.V[:,k]
        
        features = maximization_step(self.knots, self.x, self.V, c, self.ytrue) 
        feature_matrix = features.nonlinFeatureConstruction()
        
        # Construct the optimization problem
        var = cp.Variable(feature_matrix.shape[1])
        objective = cp.Minimize(cp.sum_squares(feature_matrix @ var - self.ytrue.flatten()))
        
        ## Construct constraints
        # Way 1: matrix representation
        # constraints = []
        # for icon in range(len(self.subfeaturesall)-1):
        #     subf_before = self.subfeaturesall[icon]
        #     subf_after = self.subfeaturesall[icon+1]
        #     constraints.append(subf_before @ var[:subf_before.shape[1]] - subf_after @ var[:subf_after.shape[1]]<=0)
        # Way 2: sign determination
        
        nb_free_var = self.subfeaturesall[0].shape[1]-1 # 5 for cubic spline
        constraints = [0 <=var[nb_free_var:] ]
        for icon in range(len(self.subfeaturesall)):
            f = self.subfeaturesall[icon]
            lastcol = f[:,-1].tolist()
            nb_neg = sum(1 for item in lastcol if item < 0)
            if nb_neg > 0:
                print(f'{icon+nb_free_var+1}th. coeff should be zero.')
                constraints.append(var[icon+nb_free_var]<=0)
                
        
        prob = cp.Problem(objective, constraints)
        # print("Is problem DQCP?: ", prob.is_dqcp())
        
        # The optimal results
        # print('Start solving optimization...')
        # result = prob.solve(verbose=True)
        prob.solve(solver=cp.ECOS)
        # print('Problem solved!')
        # print(constraints)
        self.b_estimate = var.value
        return var.value
