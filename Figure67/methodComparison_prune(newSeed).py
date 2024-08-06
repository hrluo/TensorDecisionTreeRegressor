#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Comparison Code
Based on the github code for the "Tensor Gaussian Process with Contraction for Multi-Channel Imaging Analysis" at https://github.com/husun0822/TensorGPST/commit/bf3cbe4ae413e170b30347f5bc83f160c125a92c
''' 
import sys
sys.path.insert(1, '../TensorGPST-main')

# built-in packages for tensor operations & optimization
import numpy as np
import pandas as pd
import prox_tv as ptv # package for solving the fused-lasso problem in the proximal step (see the installation guide at: https://github.com/albarji/proxTV)
import copy, math, sys, os, random, types, collections


# self-developed module for simulation data generation and implementing the model
from utils import *
from Simulation_Mod import Simulation # import the function for generating the simulation data
from model import TensorGPST, TensorGP # import the Tensor-GPST model and Tensor-GP model
from tensorly.regression.cp_regression import CPRegressor
from tensorly.regression.tucker_regression import TuckerRegressor


# packages for visualizing the results
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rc('image', cmap='Reds') # set default colormap as red vs. white

from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy

from TensorDecisionTreeRegressor import *
#Debugging import
import importlib
var = 'TensorDecisionTreeRegressor'
package = importlib.import_module(var)
for name, value in package.__dict__.items():
    if not name.startswith("__"):
        globals()[name] = value


# In[2]:


import time

# Start the timer
start_time = time.time()

# Your code here
for i in range(1000000):
    _ = i**2

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
print(type(end_time - start_time))
end_time - start_time


# In[5]:


sim_param = {'N': 1000, 'sigma': 0.5, 'train_ratio': 0.75}
for alpha in [0.0,0.1,0.5]:
    for md in [0,1,2,3,4,5,6]:
        n_repeat = 20
        results_train, results_test, cover_train, cover_test, times_train, times_test = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for iteration in range(n_repeat): # number of iteration = 2
            # generate simulation data
            data = Simulation(sim_param, seed = 1000+iteration)
            
            # ------------------------ #
            # - Fit tensorTree mean  - #
            # ------------------------ #
            print("Fitting tensorTree mean...")
            TT_tensor_regression_model = TensorDecisionTreeRegressor(max_depth=md, min_samples_split=2, split_method='variance',split_rank=3,n_mode=4) # split_rank=3 to match the TensorGPST empirically works well
            TT_tensor_regression_model.sample_rate = 0.1
            start_time = time.time()
            TT_tensor_regression_model.fit(data.train_X, data.train_y)
            TT_tensor_regression_model.prune(data.train_X,data.train_y,'mean',alpha)
            end_time = time.time()
            times_train['TT_mean'].append(end_time - start_time)
            times_train['TT_CP'].append(end_time - start_time)
            times_train['TT_Tucker'].append(end_time - start_time)


            TT_mean_train_pred = TT_tensor_regression_model.predict(data.train_X,regression_method='mean')

            start_time = time.time()
            TT_mean_pred = TT_tensor_regression_model.predict(data.test_X,regression_method='mean')
            end_time = time.time()
            times_test['TT_mean'].append(end_time - start_time + times_train['TT_mean'][-1])
            
            train_MSE, test_MSE = np.mean((TT_mean_train_pred - data.train_y)**2), np.mean((TT_mean_pred - data.test_y)**2)
            results_train['TT_mean'].append(train_MSE)
            results_test['TT_mean'].append(test_MSE)
            
            # ------------------------ #
            # -   Fit tensorTree CP  - #
            # ------------------------ #
            print("Fitting tensorTree CP...")
            start_time = time.time()
            TT_tensor_regression_model.fit(data.train_X, data.train_y)
            TT_tensor_regression_model.prune(data.train_X,data.train_y,'cp',alpha)
            TT_CP_train_pred = TT_tensor_regression_model.predict(data.train_X,regression_method='cp')
            end_time = time.time()
            times_test['TT_CP'].append(end_time - start_time + times_train['TT_CP'][-1])

            TT_CP_pred = TT_tensor_regression_model.predict(data.test_X,regression_method='cp')
            train_MSE, test_MSE = np.mean((TT_CP_train_pred - data.train_y)**2), np.mean((TT_CP_pred - data.test_y)**2)
            results_train['TT_CP'].append(train_MSE)
            results_test['TT_CP'].append(test_MSE)

            # ------------------------ #
            # - Fit tensorTree Tucker- #
            # ------------------------ #
            print("Fitting tensorTree Tucker...")
            start_time = time.time()
            TT_tensor_regression_model.fit(data.train_X, data.train_y)
            TT_tensor_regression_model.prune(data.train_X,data.train_y,'tucker',alpha)
            TT_Tucker_train_pred = TT_tensor_regression_model.predict(data.train_X,regression_method='tucker')
            end_time = time.time()
            times_test['TT_Tucker'].append(end_time - start_time + times_train['TT_Tucker'][-1])

            TT_Tucker_pred = TT_tensor_regression_model.predict(data.test_X,regression_method='tucker')
            train_MSE, test_MSE = np.mean((TT_Tucker_train_pred - data.train_y)**2), np.mean((TT_Tucker_pred - data.test_y)**2)
            results_train['TT_Tucker'].append(train_MSE)
            results_test['TT_Tucker'].append(test_MSE)
            print('depth=',md,'--->')
            print(f"Iteration {iteration} done.")
        print('depth=',md,'Done')
        df_train = pd.DataFrame(results_train, index=list(range(1,1+n_repeat)))
        df_test = pd.DataFrame(results_test, index=list(range(1,1+n_repeat)))
        df = pd.concat([df_train, df_test])
        # Save to CSV
        df_train.to_csv('methodPrune_train_depth_'+str(md)+'_'+str(alpha)+'.csv')
        df_test.to_csv('methodPrune_test_depth_'+str(md)+'_'+str(alpha)+'.csv')

        dft_train = pd.DataFrame(times_train, index=list(range(1,1+n_repeat)))
        dft_test = pd.DataFrame(times_test, index=list(range(1,1+n_repeat)))
        # Save to CSV
        dft_train.to_csv('methodPrune_train_depth_'+str(md)+'_time_'+str(alpha)+'.csv')
        dft_test.to_csv('methodPrune_test_depth_'+str(md)+'_time_'+str(alpha)+'.csv')


