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
        TT_Tucker_train_pred = TT_tensor_regression_model.predict(data.train_X,regression_method='tucker')
        end_time = time.time()
        times_test['TT_Tucker'].append(end_time - start_time + times_train['TT_Tucker'][-1])

        TT_Tucker_pred = TT_tensor_regression_model.predict(data.test_X,regression_method='tucker')
        train_MSE, test_MSE = np.mean((TT_Tucker_train_pred - data.train_y)**2), np.mean((TT_Tucker_pred - data.test_y)**2)
        results_train['TT_Tucker'].append(train_MSE)
        results_test['TT_Tucker'].append(test_MSE)

        # ------------------------ #
        # ----- Fit GPST-low ----- #
        # ------------------------ #
        print("Fitting GPST-low...")
        algo_params = {'N-train': math.ceil(sim_param['train_ratio'] * sim_param['N']), 
                'N-test': math.ceil((1 - sim_param['train_ratio']) * sim_param['N']),
                'Latent-Dim': (3,3),
                'Latent-Rank': (3,3,3),
                'lambda': 0.1,
                'Init-A': "random",
                'Init-B': "random",
                'Init-U': "random",               
                'sigma-init': "random",
                'plot': False,
                'seed': 2022}
        start_time = time.time()
        model = TensorGPST(param = algo_params, data = data)
        model.fit(max_iter = 2000, lr = 1e-4, tol = 1e-3, print_freq = 5000)
        end_time = time.time()
        times_train['GPST-low'].append(end_time - start_time)
        times_test['GPST-low'].append(end_time - start_time)
        train_MSE, test_MSE = np.mean((model.y_train_pred[:,0] - data.train_y)**2), np.mean((model.y_pred[:,0] - data.test_y)**2) # train/test MSE
        train_prob, test_prob = coverage_probability(model, data) # train/test coverage
        
        results_train['GPST-low'].append(train_MSE)
        results_test['GPST-low'].append(test_MSE)
        cover_train['GPST-low'].append(train_prob)
        cover_test['GPST-low'].append(test_prob)
        
        # ------------------------ #
        # ----- Fit GPST-high ---- #
        # ------------------------ #
        print("Fitting GPST-high...")
        algo_params['lambda'] = 1.0
        start_time = time.time()
        model = TensorGPST(param = algo_params, data = data)
        model.fit(max_iter = 2000, lr = 1e-4, tol = 1e-3, print_freq = 5000)
        end_time = time.time()
        times_train['GPST-high'].append(end_time - start_time)
        times_test['GPST-high'].append(end_time - start_time)
        train_MSE, test_MSE = np.mean((model.y_train_pred[:,0] - data.train_y)**2), np.mean((model.y_pred[:,0] - data.test_y)**2) # train/test MSE
        train_prob, test_prob = coverage_probability(model, data) # train/test coverage
        results_train['GPST-high'].append(train_MSE)
        results_test['GPST-high'].append(test_MSE)
        cover_train['GPST-high'].append(train_prob)
        cover_test['GPST-high'].append(test_prob)
        
        
        # ------------------------ #
        # ----- Fit GPST-Hard ---- #
        # ------------------------ #
        # GPST-hard assumes that the sparsity structure of the tensor contracting factors are known
        print("Fitting GPST-Hard...")
        algo_params['lambda'] = 0.0
        algo_params['Init-A'] = "hard"
        algo_params['Init-B'] = "hard"
        start_time = time.time()
        model = TensorGPST(param = algo_params, data = data)
        model.fit(max_iter = 2000, lr = 1e-4, tol = 1e-3, print_freq = 5000)
        end_time = time.time()
        times_train['GPST-Hard'].append(end_time - start_time)
        times_test['GPST-Hard'].append(end_time - start_time)
        train_MSE, test_MSE = np.mean((model.y_train_pred[:,0] - data.train_y)**2), np.mean((model.y_pred[:,0] - data.test_y)**2) # train/test MSE
        train_prob, test_prob = coverage_probability(model, data) # train/test coverage
        results_train['GPST-Hard'].append(train_MSE)
        results_test['GPST-Hard'].append(test_MSE)
        cover_train['GPST-Hard'].append(train_prob)
        cover_test['GPST-Hard'].append(test_prob)
        
        
        # ------------------------------- #
        # -------- Fit Tensor-GP -------- #
        # ------------------------------- #
        print("Fitting GP...")
        algo_params = {'N-train': math.ceil(sim_param['train_ratio'] * sim_param['N']), 
                'N-test': math.ceil((1 - sim_param['train_ratio']) * sim_param['N']),
                'Latent-Rank': (3,3,3),
                'plot': False,
                'seed': 2022}
        start_time = time.time()
        Tensor_GP_regression = TensorGP(param = algo_params, data = data)
        Tensor_GP_regression.fit(max_iter = 2000, lr = 1e-4, tol = np.sqrt(1e-5), print_freq = 5000)
        end_time = time.time()
        times_train['GP'].append(end_time - start_time)
        times_test['GP'].append(end_time - start_time)
        train_MSE, test_MSE = np.mean((Tensor_GP_regression.y_train_pred[:,0] - data.train_y)**2), np.mean((Tensor_GP_regression.y_pred[:,0] - data.test_y)**2) # train/test MSE
        train_prob, test_prob = coverage_probability(Tensor_GP_regression, data) # train/test coverage
        results_train['GP'].append(train_MSE)
        results_test['GP'].append(test_MSE)
        cover_train['GP'].append(train_prob)
        cover_test['GP'].append(test_prob)
        
        
        # ------------------------ #
        # -------- Fit CP -------- #
        # ------------------------ #
        print("Fitting CP...")
        start_time = time.time()
        CP_tensor_regression_model = CPRegressor(weight_rank=9, reg_W=30) # 30 empirically works well
        CP_tensor_regression_model.fit(data.train_X, data.train_y);
        CP_train_pred = CP_tensor_regression_model.predict(data.train_X)
        end_time = time.time()
        times_train['CP'].append(end_time - start_time)

        start_time = time.time()
        CP_pred = CP_tensor_regression_model.predict(data.test_X)
        end_time = time.time()
        times_test['CP'].append(end_time - start_time + times_train['CP'][-1])
        
        train_MSE, test_MSE = np.mean((CP_train_pred - data.train_y)**2), np.mean((CP_pred - data.test_y)**2)
        results_train['CP'].append(train_MSE)
        results_test['CP'].append(test_MSE)
        
        # ------------------------ #
        # ------ Fit Tucker ------ #
        # ------------------------ #
        print("Fitting Tucker...")
        start_time = time.time()
        Tucker_tensor_regression_model = TuckerRegressor(weight_ranks=[3, 3, 3], reg_W=30) # 30 empirically works well
        Tucker_tensor_regression_model.fit(data.train_X, data.train_y);
        Tucker_train_pred = Tucker_tensor_regression_model.predict(data.train_X)
        end_time = time.time()
        times_train['Tucker'].append(end_time - start_time)

        start_time = time.time()
        Tucker_pred = Tucker_tensor_regression_model.predict(data.test_X)
        end_time = time.time()
        times_test['Tucker'].append(end_time - start_time + times_train['Tucker'][-1])

        train_MSE, test_MSE = np.mean((Tucker_train_pred - data.train_y)**2), np.mean((Tucker_pred - data.test_y)**2)
        results_train['Tucker'].append(train_MSE)
        results_test['Tucker'].append(test_MSE)
    
 

        print('depth=',md,'--->')
        print(f"Iteration {iteration} done.")
    print('depth=',md,'Done')
    df_train = pd.DataFrame(results_train, index=list(range(1,1+n_repeat)))
    df_test = pd.DataFrame(results_test, index=list(range(1,1+n_repeat)))
    df = pd.concat([df_train, df_test])
    # Save to CSV
    df_train.to_csv('methodComparison_train_depth_'+str(md)+'1.csv')
    df_test.to_csv('methodComparison_test_depth_'+str(md)+'1.csv')

    dft_train = pd.DataFrame(times_train, index=list(range(1,1+n_repeat)))
    dft_test = pd.DataFrame(times_test, index=list(range(1,1+n_repeat)))
    # Save to CSV
    dft_train.to_csv('methodComparison_train_depth_'+str(md)+'_time1.csv')
    dft_test.to_csv('methodComparison_test_depth_'+str(md)+'_time1.csv')


