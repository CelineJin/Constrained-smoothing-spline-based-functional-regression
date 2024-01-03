#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:16:05 2023

@author: celinejin
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy import integrate
from maximization_step import maximization_step
from constrained_maximization_step import constrained_maximization_step
# from expectation_step import expectation_step
from getresults import getresults
from KLExpansion import KLExpansion
import math
# from expectation_step import expectation_step
from getinput import getinput
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import scipy.integrate as integrate
import pandas as pd
import os
plt.close('all')


### simulation case generation
## Case 1
# np.random.seed(134)
# x = np.random.rand(6,10)*3 # x is a n*k matrix with n representing nb of 
# t = np.linspace(0,1,num=x.shape[1])
# #observations, k representing 
# betafun_true = np.zeros((1,len(t)))
# for ib in range(len(t)):
#     betafun_true[:,ib] = math.sin(t[ib]*2*np.pi)
# a = 1 
# y = a + np.inner(x,betafun_true)*t[1] 

## Case 2
x = pd.read_csv('Xdata_1000.csv',header=None).values
# fx = plt.figure()
# for ix in range(x.shape[0]):
#     plt.plot(x[ix,:])
# plt.show()
t = np.linspace(0,1, num=x.shape[1])
inputs = getinput(t,x)
betafun_true = inputs.b_vec()
# fb = plt.figure()
# plt.plot(betafun_true)
# plt.show()
y = inputs.get_ytrue()
# fy = plt.figure()
# plt.plot(y)
# plt.show()
y_over_t = inputs.get_y_over_t()
# fyt = plt.figure()
# for iy in range(y_over_t.shape[0]):
#     plt.plot(y_over_t[iy,:])
# plt.show()




## X conversion  
# knots = np.array([0.01,0.02,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
knots = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# knots = np.linspace(0.01,0.9,7)
MAE = []
# instance = KLExpansion(x)
# kuv = instance.cov()

# instance.eigendecomposition()
# K_x = instance.determineK_x()
# D = instance.D
# V = instance.V

## Use matlab results
# D = pd.read_csv('D_matlab.csv',header=None).values
V = pd.read_csv('V_1000_matlab.csv',header=None).values
agg_level = len(os.listdir('data'))

def calprojection(x,K_x,V,delt):
    cik = np.zeros((x.shape[0],K_x))
    for i in range(x.shape[0]):
        for k in range(K_x):
            f = np.multiply(delt, V[:,k])
            cik[i,k] =  x[i,:]@f
    return cik

def findBestKx(K_x,V,t,knots,x,y, bestKx):
    # print(f'Kx = {K_x}')
    Phi = V[:,0:K_x]
    # delt = t[1]*np.ones((V.shape[0],1)).flatten()
    delt = np.ones((V.shape[0],1)).flatten()
    
    # Kb = len(knots)
    
    # ### Unconstrained spline training...
    # c = calprojection(x,K_x,Phi,delt)
    # to_max = maximization_step(knots,x, Phi, c, y)  
    # # X = to_max.linFeatureConstruction()  # Spline basis 1
    # to_max.nonlinFeatureConstruction()  # Spline basis 2: cubic spline
    # b_est, ypred = to_max.maximization()
    
    ### Constrained spline training ...
    to_max = constrained_maximization_step(knots, x, Phi, y)
    to_max.get_subfeatures()
    b_est = to_max.constrained_least_square().reshape(1,-1)
    
    if K_x == bestKx:
        print(f'best Kx = {bestKx}')
        print(f'b estimates: {b_est}')
    # print(f'b estimates: {b_est}')
    ## fitting error:
    # mae = np.mean(np.abs(ypred-y))
    
    ## cal error:
    result = getresults(x, t, b_est, knots, y, agg_level)
    result.get_bvalues()
    beta_values = result.beta_pred
    ycal = result.get_ypred()
    ypred_over_t, ypred_agg_t = result.get_ypred_over_t()
    rmse, mae, perc_err = result.get_errors()
    return mae, b_est, beta_values, ycal, ypred_over_t, ypred_agg_t

bestKx = 10000
for K_x in range(1,10):
    print(f'Kx = {K_x}')
    results = findBestKx(K_x, V, t, knots, x, y, bestKx)
    mae = results[0]
    MAE.append(mae)
    
MAE_num = np.array(MAE)
f1 = plt.figure()
plt.plot(MAE_num,'.')
plt.title('MAE')
plt.show()

K_x = np.argmin(MAE_num)+1
mae = np.min(MAE_num)
perc_err = mae/np.mean(np.abs(y))  
print(f'Kx = {K_x}, mae = {mae}, perc_err = {perc_err}')


### Report final result
bestKx = K_x
final_results = findBestKx(K_x,V,t,knots,x,y, bestKx)
beta_values = final_results[2]
ycal = final_results[3]
ypred_over_t = final_results[4]
ypred_agg_t = final_results[-1]

f2 = plt.figure()
plt.plot(t, betafun_true.flatten(), 'b-')
plt.plot(t,beta_values.flatten(),'r-')
plt.title('beta function')
plt.legend(['real beta function','estimated beta function'])
plt.show()

f3 = plt.figure()
plt.plot(y, 'b-')
plt.plot(ycal, 'r-')
plt.title('y')
plt.legend(['real y','predicted y'])
plt.show()

f4 = plt.figure()
for iobs in range(ypred_over_t.shape[0]):
    plt.plot(np.linspace(0,1,ypred_over_t.shape[1]),ypred_over_t[iobs,:])
plt.title('predicted y overtime')
plt.savefig('predicted_y_overtime_uncon.png')

# for iobs in range(ypred_agg_t.shape[0]):
#     plt.figure()
#     plt.plot(np.linspace(0,1,y_over_t.shape[1]),y_over_t[iobs,:])
#     plt.plot(np.linspace(0,1,ypred_over_t_uncon.shape[1]), ypred_over_t_uncon[iobs,:])
#     plt.plot(np.linspace(0,1,ypred_agg_t.shape[1]),ypred_agg_t[iobs,:])
    
#     plt.legend(['true y(t)','y(t)-constrained prediction','y(t)-unconstrained prediction'])
#     plt.title(f'y(t) of Sample {iobs+1}')
#     plt.xlabel('t')
#     plt.savefig(f'y_overtime_Sample{iobs+1}.png')




# ## Run maximization and expectation until the difference of two consecutive error (after expectation) is less than epsilon
# error = 10000
# epsilon_error = (np.mean(abs(y))*0.1)**2
# i = 1
# mse = []
# while (error > epsilon_error) & (i<1):
#     print(f'Learning round {i}')
#     print('####################################')
    
#     ## expectation step - train for knots, given b_est
#     print('Expectation starts...')
#     initial_guess = knots
#     bounds = ((0,1),)
#     for ik in range(len(knots)-1):
#         bounds = bounds + ((0,1),)
        
#     # to_exp = expectation_step(t, b_est, X, c, Phi, y, initial_guess, bounds)
#     # result = to_exp.optimization()
        
#     def objective(x, t, b_est, M, c, Phi, y): # M is the X matrix out of the maximization step
#         # X = M
#         nsample = M.shape[0]
#         Kb = len(x)
#         obj = b_est[:,0] #alpha
#         temp = b_est[:,1]*M[:,1].reshape(-1,1)
#         temp += obj
#         obj = temp
#         obj += b_est[:,2]*M[:,2].reshape(-1,1)
#         Mk = np.zeros((nsample,Kb))
#         for k in range(Kb):  
#             for i in range(nsample):
#                 fac_Mk = np.matmul(Phi.transpose(), np.maximum(t-x[k],0)).reshape(-1,1)
#                 Mk[i,k] = np.matmul(c[i,:],fac_Mk)
#             obj += b_est[:,k+2]*Mk[:,k].reshape(-1,1)
#         return np.mean((y-obj)**2)
    
    
#     # result = minimize(objective, x0=initial_guess, args = (t,b_est,X,c,Phi,y), method='L-BFGS-B', bounds = bounds)
#     result = basinhopping(objective, x0=initial_guess, minimizer_kwargs = {'args':(t,b_est,X,c,Phi,y), 'bounds':bounds})
#     knots = result.x
#     print('Expectation ends.')
#     # print(f'Knots are {knots}')
#     # print('####################################')
    
#     ## maximization step - train for b_est, given knots
#     print('Maximization starts...')
#     to_max = maximization_step(knots,x, Phi, c, y)  
#     X = to_max.featureConstruction()  
#     b_est, ypred = to_max.maximization()
#     print('Maximization ends')
#     # print(f'b estimates are {b_est}')
#     error = np.mean((ypred-y)**2)
#     mse.append(error)
#     print(f'y pred are {ypred}')
#     print(f'MSE is {error}, knots are {knots}, b estimates are {b_est}')
#     i += 1
    
# final_knots = knots

### Generate estimate and prediction
# result = getresults(x, t, b_est, final_knots, ypred, y)
# result.get_bvalues()
# b_values = result.beta_pred
# ycal = result.get_ypred()

# crmse, cmae, cperc_err = result.get_errors()
# print(f'Cal results: mae = {cmae}, perc_err = {cperc_err}')
