#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:43:34 2023

@author: celinejin
"""

import numpy as np
import scipy

class KLExpansion:
    def __init__(self, x):
        self.x = x
        self.D = None
        self.V = None
        self.K_x = None
        self.cik = None
        self.kuv = None
        # self.cov()
        
        
        
    def cov(self):
        kuv = np.zeros((self.x.shape[1],self.x.shape[1]))
        for u in range(self.x.shape[1]):
            xu_bar = np.mean(self.x[:,u])
            for v in range(self.x.shape[1]):
                xv_bar = np.mean(self.x[:,v])
                Kuv = (self.x[:,u]-xu_bar)@(self.x[:,v]-xv_bar)
                kuv[u,v] = Kuv/self.x.shape[0]
        self.kuv = kuv  # remove the nuget when handling real data 
        return kuv
    
    def eigendecomposition(self):
        covmat = self.cov() 
        covmat = (covmat+covmat.transpose())/2
        # D, V = np.linalg.eig(covmat)
        D, V = scipy.linalg.eigh(covmat)
        if np.all(D > 0):
            print('The matrix is pos. def.')
            self.D = D
            self.V = V
        else:
            print('The matrix is not pos. def. It is being adjusted...')
            covmat += 0.0001*np.identity(covmat.shape[0])
            # D, V = np.linalg.eig(covmat)
            D, V = scipy.linalg.eigh(covmat )
            self.D = D
            self.V = V
            
        return D, V
    
    def determineK_x(self):
        K_x = len(self.D)
        eps = 0.001
        for D_i in range(len(self.D)-1):
            d1 = self.D[D_i]
            d2 = self.D[D_i+1]
            if abs(d1-d2) < eps:
                K_x = D_i
        self.K_x = K_x
        return K_x
    
    def calprojection(self):
        cik = np.zeros((self.x.shape[0],self.K_x))
        for i in range(self.x.shape[0]):
            for k in range(self.K_x):
                cik[i,k] =  self.x[i,:]@self.V[:,k]
        self.cik = cik
        return cik
        
