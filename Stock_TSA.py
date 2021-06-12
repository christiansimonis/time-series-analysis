""" 1) Stock_TSA - Approach: TSA models a yearly growth rate, combined with probabilistic model."""
# Disclaimer: There have been several attempts to predict financial markets and stock prices using time series analysis. Many of them were not successful!
# Neither trading nor investment decisions should be influenced by this repository and the code, which is built only to introduce and demonstrate a methodology for time series modeling.
# No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions!
#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2021"
__version__ = "2.1"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"

# Approach: TSA models a yearly return rate, combined with a probabilistic model.
# While the general growth rate of the stock or index is described in a domain model,
# especially non-efficient artifacts are modeled in a probabilistic way, 
# including these parts that the domain model is not capable of describing. 
# Assumptions rely on the course tinancial markets by Robert Shiller. 
# Information links (no promotion), see sources: 
# https://www.coursera.org/learn/financial-markets-global (no promotion)
# and https://en.wikipedia.org/wiki/Brownian_model_of_financial_markets

 
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
from finquant.portfolio import build_portfolio, EfficientFrontier  
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
import warnings                                                    # https://docs.python.org/3/library/warnings.html
import random                                                      # https://docs.python.org/3/library/random.html
import datetime as dt                                              # https://docs.python.org/3/library/datetime.html
#-----------------------------------------------------------------------------------------------------------------------------------
# Hint: No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions
#----------------------------------------------------------------------------------------------------------------------------------- 
# Class definition TSA
#-----------------------------------------------------------------------------------------------------------------------------------
 
class Stock_Analysis:
    """The purpose of the class Stock_Analysis is:
        - to model and predict the time series behavior
        - to visualize the model results 
    """


    #Initialization
    def __init__(self):
        """initial call""" 
        
    #Get data via YFIN API
    def obtain_timeseries(self,Stock_Name, start):
        #--------------------------------------------------------
        """ obtain timeseries for stocks
        e.g. --> obtain_timeseries("AAPL","2018-07-20")
        
        Input: 
                Stock_Name:     Name of stock, e.g. "AAPL"
                start:          Start data, from which stock data should be downloaded, e.g. "2018-07-20"
    
    
        Output:
                Time:           Time data as numpy.ndarray
                Stock:          Stock data as numpy.ndarray
                Time_idx:       Time for user visualization: Raw time data as pandas.core.indexes.datetimes.DatetimeIndex
        
        Class:
                 DF:            Dataframe, consisting of time and closing price information
        """
            
        #ldownload with Yahoo Finance API
        
        if  hasattr(self,"end") == False:
            stocks = yfin.download(Stock_Name, start) # till most recent value
        else:
            stocks = yfin.download(Stock_Name, start, end = self.end) # till definition
        
        stocks.columns = stocks.columns.to_flat_index()
       
        
        #Export time series of stock sequence
        Stock = stocks.loc[:, "Close"].to_numpy()
        Time = stocks.loc[:, "Close"].index.to_numpy().astype("float")
        Time_idx = stocks.loc[:, "Close"].index
        self.DF = pd.DataFrame({   'ds': Time_idx,
                                    'y': stocks.loc[:, "Close"]})
        return Time, Stock, Time_idx 
        
    
    #Conduct train / test split based on target by user
    def conduct_train_test_split(self,option,split_factor, Time, Stock):   
        #--------------------------------------------------------
        """ predictics, using forecast model, consisting of a domain model and a data-driven model
        e.g. --> conduct_train_test_split(1,0.3, np.array([1, 2, 3]))
        
        Input: 
                option:         User choice: #1 = real prediction , 2= backtest
                split_factor:   Train test split
                Time:           Time data
                Stock:          Stock data
    
            
        Output:
                Time_training:  Time allocated to Training data
                y_training:     Labels allocated to Training data
                Time_test:      Time allocated to Test data
                y_test:         Labels allocated to Test data  
        """
        
        if option == 1: #Option 1) Real forecast
            delta_T = Time[1]-Time[0]
            Label_span = Time.max()-Time.min()
        
            # Chosing Training data proportional split factor
            Time_training = Time.copy()
            y_training = Stock.copy()
            
            
            #take most recent data as test data
            Time_test = np.arange(Time.max(),Time.max()  + (1-split_factor)*Label_span, delta_T)
            y_test = []
            
        
        
        else: #Option 2) Simulate real forecast (done in past)
            length = len(Time)
            till = int(np.round(split_factor*length))
            
            # Chosing Training data proportional split factor
            Time_training = Time[0:till]
            y_training = Stock[0:till]
            
            
            #take most recent data as test data
            Time_test = Time[till+1:length]
            y_test = Stock[till+1:length]
        
        return Time_training, y_training, Time_test, y_test
    
    

        
    #domain model for times series description in stock market
    def func(self, x, a, c):
        #--------------------------------------------------------
        """ Domain model to describe exponential behavior
        e.g. --> func(np.array([1, 2, 3]),7,8,9)
        
        Input: 
                x:             Input
                a:             Scaling factor, multiplied to exp-function
                b:             interest parameter --> can be specified by user to incorporate domain knowledge
                c:             Constant offset parameter
        
        Output:
                y:             Ouput according to domain model
             
        """
        
        #User choice, representing market knowledge
        if hasattr(self,"exp_interest") == False:
            b = 1.07 #interest in the long run, e.g. 1.07 = 7% interest
        else:
            b = self.exp_interest #otherwise, take class attribute
        
        #Calculation of domain model
        y = a * np.exp(b * x) + c
        return y
    
    #Forecasting model
    def fit_ForecastMdl(self, X,Label,L,N):
        #--------------------------------------------------------
        """ fits forecast model to data, using a domain model, combined with a data-driven model
        e.g. --> fit_ForecastMdl(np.array([[1,4]]).T,np.array([1,5]).T,2,2)
        
        Input: 
                 X:              Feature as Input, 
                 Label:          Ground Truth as label to be learned (output)
                 L:              Length scale: Hyperparameter of Gaussian process, kernel definition
                 N:              restarts of optimizer: Hyperparameter of Gaussian process fitting
            
        Output:
                 forecast:      Forecast model regression value
                 sigma:         Uncertainty, represented by standard deviation
                 y:             Domain regression value
                 
        Class:
                 reg:           Domain regression model (as part of class)
                 gpr:           Gaussian process model (as part of class)
                
        """
        # Domain model, e.g. via exponential approach (alternative linear model in bracket comments)
        
        #Exp function fitting
        reg, pcov = curve_fit(self.func, X[:,0], Label) #fit of domain model, Alternative: #reg = LinearRegression().fit(Time_scaled, Label)
    
        #Exp function evaluation
        y = self.func(X[:,0], *reg)  #evaluation of domain model, Alternative:  #y = reg.predict(Time_scaled) #linear function
    
        
        
        #Calculation of Residuum
        res = Label.copy() - y.copy() #exp function
        sigma_est = np.std(res)*15 #safety margin
        
        #Definition of Machine Learning model to learn residuum in supervised manner
        kernel = 1.0 * RBF(length_scale=L, length_scale_bounds=(L*1e-1, L*1e1)) + 1e3*WhiteKernel(noise_level=1e2*sigma_est, noise_level_bounds=(1e1*sigma_est, 1e2*sigma_est)) # Alternative: #kernel = 1.0 * RationalQuadratic(length_scale=L) + WhiteKernel(noise_level=0, noise_level_bounds=(1e-8, sigma_est)) 
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=N,alpha = 0.5)
        gpr.fit(X, res)
        
        
        #Fit of Machine Learning model
        GP_pred, sigma = gpr.predict(X, return_std=True) 
        
        #Combination of results
        forecast = GP_pred + y
        self.gpr = gpr # data-driven (probabilistic) model
        self.reg = reg # domain model
        
        return forecast, sigma, y
     
        
    #Prediction function
    def pred_ForecastMdl(self, X):
        #--------------------------------------------------------
        """ predictics, using forecast model, consisting of a domain model and a data-driven model
        e.g. --> forecast, sigma, y = fit_ForecastMdl(np.array([[1,4]]).T,np.array([1,5]).T,2,2); pred_ForecastMdl(np.array([[1,1.2]]).T)
        
        Input: 
            X:              Feature as Input, 
            reg:            Domain regression model
            gpr:            Gaussian process model
            
        Output:
             forecast_pred: Predicted forecast model regression value
             sigma_pred:    Predicted uncertainty, represented by standard deviation
             y_pred:        Predicted domain regression value       
        """
        
        #predict with  domain model
        y_pred = self.func(X[:,0], *self.reg) #exp function, Alternative: #y_pred = reg.predict(Time_scaled) # linear function
        
        #predict with data-driven model
        GP_pred, sigma_pred = self.gpr.predict(X, return_std=True) 
        
        #Combine predictions
        forecast_pred = GP_pred + y_pred
        
        return forecast_pred, sigma_pred, y_pred

    #Visualization
    def vis_results(self,Time_training,forecast,sigma,Time_test,forecast_future,sigma_future, Time,Stock, Stock_Name):
        #--------------------------------------------------------
        """ visualizes results of forecast model, consisting of a domain model and a data-driven model
        e.g. --> runfile('RUN_Stock-Forecast.py')
        
        Input: 
            Time_training:   Time allocated to Training data
            forecast:        Forecast model regression value
            sigma:           Uncertainty, represented by standard deviation
            Time_test:       Time allocated to Test data
            forecast_future: Predicted forecast model regression value
            sigma_future:    Predicted uncertainty, represented by standard deviation
            Time:            Time data as numpy.ndarray
            Stock:           Stock data as numpy.ndarray
            Stock_Name:      Name if Stock or Index
        """
        
        #Fit & Prediction visualization of TSA (Time series analysis) approach
        plt.style.use("seaborn")
        plt.plot(Time_training,forecast,'b-',linewidth=3, label = 'Model Fit')
        plt.fill(np.concatenate([Time_training, Time_training[::-1]]),np.concatenate([forecast - 3 * sigma,(forecast + 3 * sigma)[::-1]]),
                 alpha=.3, fc='y', ec='None', label='99% confidence interval for training')
        plt.plot(Time_test,forecast_future,'k-.',linewidth=2, label = 'Forecast with Prediction Model')
        plt.fill(np.concatenate([Time_test, Time_test[::-1]]),np.concatenate([forecast_future - 3 * sigma_future,(forecast_future + 3 * sigma_future)[::-1]]),
                 alpha=.2, fc='g', ec='None', label='99% confidence interval for prediction')
        plt.scatter(Time,Stock, label = Stock_Name, c="coral")
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Closing Price', fontsize=16)
        plt.legend(loc='upper left', shadow=False, ncol=1)
        return 0
    
    
    
    #Optimize portfolio using finquant library
    def optimize_pf(self, df_data, nr_mc, risk_free_rate):   
    #--------------------------------------------------------
        """ optimizes portfolio (either historic or predictive) based on defined criteria
        e.g. --> runfile('RUN_Pred-optimization.py')
        
        Input: 
                df_data:            Portfolio dataframe to be optimized
                nr_mc:              Number of samples for Monte Carlo simulation
                risk_free_rate:     risk free rate
    
            
        Output:
                opt_w:              Optimized weights for asset allocation 
        """
    
        plt.style.use("seaborn-darkgrid")
        # set line width
        plt.rcParams["lines.linewidth"] = 2
        # set font size for titles
        plt.rcParams["axes.titlesize"] = 14
        # set font size for labels on axes
        plt.rcParams["axes.labelsize"] = 12
        # set size of numbers on x-axis
        plt.rcParams["xtick.labelsize"] = 10
        # set size of numbers on y-axis
        plt.rcParams["ytick.labelsize"] = 10
        # set figure size
        plt.rcParams["figure.figsize"] = (10, 6)
        
  
        # building a portfolio by providing stock data
        pf = build_portfolio(data=df_data)
        pf.risk_free_rate = risk_free_rate # risk free rate
        print(pf)
        pf.properties()
        
        
        # if needed, change risk free rate and frequency/time window of the portfolio
        print("pf.risk_free_rate = {}".format(pf.risk_free_rate))
        print("pf.freq = {}".format(pf.freq))
        
        """
        pf.ef_minimum_volatility(verbose=True)
        
        
        # optimisation for maximum Sharpe ratio
        pf.ef_maximum_sharpe_ratio(verbose=True)
        
        
        # minimum volatility for a given target return of 0.26
        pf.ef_efficient_return(0.26, verbose=True)
        """
        
        # optimisation for maximum Sharpe ratio
        pf.ef_maximum_sharpe_ratio(verbose=True)
        
        
        # Monte Carlo portfolios and Efficient Frontier solutions
        opt_w, opt_res = pf.mc_optimisation(num_trials=nr_mc)
        pf.mc_properties()
        pf.mc_plot_results()
        
        
        # visualization
        pf.ef_plot_efrontier()
        pf.ef.plot_optimal_portfolios()
        pf.plot_stocks()
        plt.show()
        
         
        #rprovide result
        self.optimized_weights = opt_w
        self.optimized_weights.head()
        return opt_w


#-----------------------------------------------------------------------------------------------------------------------------------

