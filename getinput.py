#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:23:41 2023

@author: celinejin
"""

import math
import numpy as np

class getinput:
    def __init__(self, t, x):
        self.t = t 
        self.x = x
        self.bfun_vec = None
        # self.bfun1()
        # self.xfun()
        # self.obs = None
        self.y = None
        self.y_over_t = None
        
    # # @staticmethod
    # def bfun1(t_var): # t in [0,1]
    #     b = 0
    #     for bj in range(1,1001):
    #         b += bj**(-4)*2**0.5*math.cos(bj*math.pi*t_var)
    #         # b += 2
    #     return b
    
    def b_vec(self):
        bfun_vec = np.zeros((1,len(self.t)))
        for ib in range(len(self.t)):
            b = 0
            for bj in range(1,1001):
                b += bj**(-4)*2**0.5*math.cos(bj*math.pi*self.t[ib])
            bfun_vec[:,ib] = -b+2
        self.bfun_vec = bfun_vec
        return bfun_vec
    
            
    def get_ytrue(self):    
        y = np.inner(self.x,self.bfun_vec)*self.t[1]
        self.y = y
        return y
    
    def get_y_over_t(self):
        t = np.linspace(0,1,self.x.shape[1])
        y_over_t = np.zeros((self.x.shape[0],self.x.shape[1]))
        for it in range(len(t)):
            y_over_t[:,it] = (np.inner(self.x[:,:it+1], self.bfun_vec[:,:it+1])*self.t[1]).transpose()
        self.y_over_t = y_over_t
        return y_over_t
        
            
