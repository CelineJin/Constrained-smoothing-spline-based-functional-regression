#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:21:15 2023

@author: celinejin
"""
import numpy as np

class getresults():
    def __init__(self, x, t, b_est, knots,y,agg):
        self.x = x
        self.t = t
        self.b_est = b_est
        self.knots = knots
        self.y = y
        self.beta_pred = None
        self.mse = None
        # self.rmse = None
        self.mae = None
        self.perc_err = None
        self.ycal = None
        self.agg = agg
        
        # self.b_func_est()
        
        

    def get_bvalues(self):
        beta_pred = np.zeros((1,len(self.t)))
        for it in range(len(self.t)):
            beta_func = self.b_est[:,0]
            beta_func += self.b_est[:,1]*self.t[it]
            for k in range(len(self.knots)):
                beta_func += self.b_est[:,k+1]*max(self.t[it]-self.knots[k],0)
            beta_pred[:,it] = beta_func
        self.beta_pred = beta_pred
        return beta_pred
    
    def get_ypred(self):
        # ycal = np.inner(self.x, self.beta_pred) * self.t[1] + self.b_est[:,0]
        ycal = np.inner(self.x, self.beta_pred) * self.t[1]
        self.ycal = ycal
        return ycal
    
    def get_ypred_over_t(self):
        t = np.linspace(0,1,self.x.shape[1])
        ypred_over_t = np.zeros((self.x.shape[0],self.x.shape[1]))
        for it in range(len(t)):
            ypred_over_t[:,it] = (np.inner(self.x[:,:it+1], self.beta_pred[:,:it+1])*self.t[1]+self.b_est[:,0]).transpose()
        agg_idx = np.linspace(0,self.x.shape[1],self.agg+1)-1
        agg_idx[0] = agg_idx[0]+1
        agg_idx.tolist()
        agg_idx = [int(item) for item in agg_idx]
        ypred_agg_t = ypred_over_t[:,agg_idx]
        return ypred_over_t, ypred_agg_t
    
    def get_errors(self):
        mse = np.mean((self.y-self.ycal)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs(self.y-self.ycal))
        perc_err = np.mean(np.abs(self.y-self.ycal)/np.abs(self.y))
        # perc_err = mae/np.mean(abs(self.y))
        # self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.perc_err = perc_err
        return rmse, mae, perc_err
            
        
            
            
            
            
            
            
            
            
