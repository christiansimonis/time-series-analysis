""" RUN_Stock_historic_TSA: evaluates the performance of TSA class based on historic data, using the index Nasdaq 100. """
# Disclaimer: There have been several attempts to predict financial markets using time series analysis. Many of them were not successful!
# Neither trading nor investing decisions should be influenced by this repository and the code, which is built only to introduce and demonstrate a methodology for time series modeling.
# No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions

#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"
# For prediction, TSA models a yearly growth rate, combined with a probabilistic model.
# While the general growth rate of the stock or index is described in a domain model,
# especially non-efficient artifacts are modeled in a probabilistic way, 
# including these parts that the domain model is not capable of describing. 
# Assumptions rely on course Financial markets by Robert Shiller. 
# Information links (no promotion), see sources: 
# https://www.coursera.org/learn/financial-markets-global (no promotion)
# and https://en.wikipedia.org/wiki/Brownian_model_of_financial_markets
# To download relevant market data, yfinance library is used.
#-----------------------------------------------------------------------------------------------------------------------------------
# Name                                Version                      License  
# FinQuant                            0.2.2                        MIT License,                                 Copyright (C) 2019 Frank Milthaler:https://github.com/fmilthaler/FinQuant/blob/master/LICENSE.txt
# numpy                               1.19.5                       BSD,                                         Copyright (c) 2005-2020, NumPy Developers: https://numpy.org/doc/stable/license.html#:~:text=Copyright%20(c)%202005%2D2020%2C%20NumPy%20Developers.&text=THIS%20SOFTWARE%20IS%20PROVIDED%20BY,A%20PARTICULAR%20PURPOSE%20ARE%20DISCLAIMED.
# yfinance                            0.1.59                       Apache License, Version 2.0,                 Copyright (c) January 2004, Ran Aroussi: https://github.com/ranaroussi/yfinance
# matplotlib                          3.4.2                        Python Software Foundation License,          Copyright (c) 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team: https://matplotlib.org/stable/users/license.html
# scikit-learn                        0.23.1                       BSD 3-Clause License,                        Copyright (c) 2007-2021 The scikit-learn developers: https://github.com/scikit-learn/scikit-learn/blob/main/COPYING  
# pandas                              1.2.4                        BSD 3-Clause License                         Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team: https://github.com/pandas-dev/pandas/blob/master/LICENSE
# seaborn                             0.11.1                       BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2012-2021, Michael L. Waskom: https://github.com/mwaskom/seaborn/blob/master/LICENSE
# scipy                               1.5.2                        BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers: https://github.com/scipy/scipy/blob/master/LICENSE.txt
# neuralprophet                       0.2.7                        MIT License,                                 Copyright (c) 2020 Oskar Triebe: https://github.com/ourownstory/neural_prophet/blob/master/LICENSE
#-----------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                             
import yfinance as yfin                                        
import matplotlib.pyplot as plt                               
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.preprocessing import MinMaxScaler                
from sklearn.linear_model import LinearRegression             
import pandas as pd                                           
import seaborn as sns                                         
from scipy.optimize import curve_fit                          
import warnings                                               # https://docs.python.org/3/library/warnings.html
import random                                                 # https://docs.python.org/3/library/random.html
import datetime as dt                                         # https://docs.python.org/3/library/datetime.html
#-----------------------------------------------------------------------------------------------------------------------------------


#General parameters
Stock_Name = ["^NDX"]   # NASDAQ 100 taken from Yahoo Finance ticker
start="2015-06-09"      # start date, from which data is loaded till today
L = 1.1                 # Length scale as GP kernel parameter
N = 2                   # Number of GP optimization runs
split_factor = 0.65     # Train test split
option = 2              # User coice: 1 = real prediction , 2 = backtest


#Loop parameters for analysis of constant prediction horizon
horizon = 720           # in days
steps = 120             # in days
show_plots = True       # show plots in loop for prediction horizon
#-----------------------------------------------------------------------------------------------------------------------------------
# Hint: No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions
from Stock_TSA import Stock_Analysis 

#----------------------------------------------------------------------------------------------------------------------------------- 
# Evaluation of historic market predictions
#-----------------------------------------------------------------------------------------------------------------------------------
 

#Start program by downloading stock data 
warnings.filterwarnings("ignore")
random.seed(0)
Evaluation = Stock_Analysis() #class call
Time, Stock,Time_idx  = Evaluation.obtain_timeseries(Stock_Name,start)


#conduct train test split
Time_training, y_training, Time_test, y_test = Evaluation.conduct_train_test_split(option,split_factor, Time,Stock)

#Scale Time Space accordingly
scaler = MinMaxScaler()
scaler.fit(Time_training.reshape(-1,1)) # Fit on training data
Time_training_scaled = scaler.transform(Time_training.reshape(-1,1))
Time_test_scaled = scaler.transform(Time_test.reshape(-1,1))






#Train on past data
forecast, sigma, y_mdl = Evaluation.fit_ForecastMdl(Time_training_scaled, y_training,L,N)

#Prediction
forecast_future, sigma_future, y_mdl_future = Evaluation.pred_ForecastMdl(Time_test_scaled)

#Visualization
Evaluation.vis_results(Time_training,forecast,sigma,Time_test,forecast_future,sigma_future, Time,Stock, Stock_Name)


if option == 0: 
    plt.title('Time series analysis', fontsize=16)
else:
    plt.title('Time series forecast', fontsize=16)
plt.show()


#Evaluation only in case of Backtest
if option == 2: 

    #Residuum analysis: Label - Model

    #Training residuum
    residuum_train = y_training - forecast
    res_train = pd.DataFrame({'y_training':       y_training, 
                                'forecast':          forecast })
    #Test residuum
    residuum_test = y_test - forecast_future
    res_test = pd.DataFrame({'y_test':            y_test, 
                             'forecast_future':   forecast_future })
                     
 
    
    #Residuum Distribution
    sns.distplot(residuum_train,kde=True,label = 'Training residuum')
    sns.distplot(residuum_test,kde=True,label = 'Test residuum')
    plt.xlabel(' Label - Model prediction', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Residuum Distribution', fontsize=16)
    plt.show()
    
    
    
    #Performance of model on Training and Test data
    
    #Training data
    sns.set_theme(style="white")
    heatmap = sns.JointGrid(data=res_train, x="y_training", y="forecast", space=0)
    heatmap.plot_joint(sns.kdeplot,fill=True, thresh=0, levels=100, cmap="RdBu_r")
    heatmap.plot_marginals(sns.histplot, color="b", alpha=1, bins=25)
    plt.xlabel('Model prediction', fontsize=16)
    plt.ylabel('Label', fontsize=16)
    plt.title('Training data', fontsize=16)
    plt.show()
                           

    # Test data
    sns.set_theme(style="white")
    heatmap = sns.JointGrid(data=res_test, x="y_test", y="forecast_future", space=0)
    heatmap.plot_joint(sns.kdeplot,fill=True, thresh=0, levels=100, cmap="RdBu_r")
    heatmap.plot_marginals(sns.histplot, color="b", alpha=1, bins=25)
    plt.xlabel('Model prediction', fontsize=16)
    plt.ylabel('Label', fontsize=16)
    plt.title('Test data', fontsize=16)
    plt.show()
    
    
    
    '''   
    #Residuum dependent on Time as Feature
    sns.scatterplot( Time_training, residuum_train,label = 'Training residuum')
    sns.scatterplot( Time_test, residuum_test,label = 'Test residuum')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Model prediction', fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.00), shadow=False, ncol=1, fontsize=16)
    plt.title('Residuum dependent on Time as Feature', fontsize=16)
    plt.show()
    '''


#Stock analysis with a constant prediction horizon with naive prediction as benchmark
#-----------------------------------------------------------------------------------------------------------------------------------
  

Const_pred_evaluation = Stock_Analysis() #class call
Time, Stock,Time_idx  = Const_pred_evaluation.obtain_timeseries(Stock_Name,start)
    
#conduct train test split
split_factor = 0.2 # Train test split
option = 2           #always Backtest analysis: #1 = real prediction , 2= backtest
Time_training, y_training__, Time_test, y_test__ = Const_pred_evaluation.conduct_train_test_split(option,split_factor, Time, Stock)
time = np.append(Time_training, Time_test)
time_scaled = np.append(Time_training_scaled, Time_test_scaled).reshape(-1,1)
y = np.append(y_training,y_test)


#initialize arrays for constant prediction analysis
const_pred = np. array([])
const_sigma = np. array([])
const_time = np. array([])
const_time_sc = np. array([])
naive_benchmark = np. array([])
target = np. array([])





#Looping with steps to calculate predictions with constant horizon
for i in range(len(Time_training),len(time)-1-horizon,steps):
    
    
    
    #solve this iteration
    try:

       #count one increment -->  index per iteration
        Time_training_sc = time_scaled[0:i].reshape(-1, 1)
        Time_test_sc = time_scaled[i+1:].reshape(-1, 1)
        
        Time_test_c = time[i+1:].reshape(-1, 1)
        y_training_c = y[0:i]
        y_test_c = y[i+1:].reshape(-1, 1)
        
        
        
        #Train on past data
        forecast_c, sigma_c, y_mdl  = Const_pred_evaluation.fit_ForecastMdl(Time_training_sc, y_training_c ,L,N=10)
    
        #Prediction, evaluated at defined prediction horizon
        forecast_future_c, sigma_future_c, y_future_c = Const_pred_evaluation.pred_ForecastMdl(Time_test_sc)
    
    
        #Fit & Prediction visualization: only if show_plots == TRUE
        if show_plots: 
            
            #Visualization
            plt.style.use("seaborn")
            Const_pred_evaluation.vis_results(Time_training_sc,forecast_c,sigma_c,Time_test_sc,forecast_future_c,sigma_future_c, Time_training_sc,y_training_c, Stock_Name)
            plt.show()

    
        #append values for constant prediction horizon
        const_pred = np.append(const_pred,forecast_future_c[horizon])
        const_sigma = np.append(const_sigma,sigma_future_c[horizon])
        const_time = np.append(const_time,Time_test_c[horizon])
        target = np.append(target,y_test_c[horizon])
        const_time_sc = np.append(const_time_sc,Time_test_sc[horizon])
        naive_benchmark = np.append(naive_benchmark,y_training_c[-1])# use naive benchmark
        
    except ValueError:
        print("Error for this iteration")
    
    


#Visualization of value-add compared to naive prediction
plt.scatter(pd.DatetimeIndex(Time),Stock, label = Stock_Name)
plt.plot(pd.DatetimeIndex(const_time),const_pred,'ko-.',linewidth=4, label = 'Prediction with constant horizon')
plt.plot(pd.DatetimeIndex(const_time),naive_benchmark,'co-.',linewidth=4, label = 'Naive prediction as benchmark')
plt.plot(pd.DatetimeIndex(const_time),target,'ro',linewidth=6, label = 'Actual target at evaluation point')
plt.xlabel('Time', fontsize=16)
plt.ylabel('Closing Price', fontsize=16)
plt.title('Time series analysis with constant prediction horizon of {} days'.format(horizon), fontsize=16)
plt.fill(np.concatenate([pd.DatetimeIndex(const_time), pd.DatetimeIndex(const_time)[::-1]]),np.concatenate([const_pred - 3 * const_sigma,(const_pred + 3 * const_sigma)[::-1]]),
         alpha=.2, fc='b', ec='None', label='+/- 3 $\u03C3$ confidence interval for prediction')
plt.legend(loc='upper left', shadow=False, ncol=1)
plt.show()  

#-----------------------------------------------------------------------------------------------------------------------------------


