""" 2.2) RUN_Stock-Forecast: demonstration of forecasting methodology, using the class Stock_TSA and Neural Prophet. """
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
# Assumptions rely on the course financial markets by Robert Shiller. 
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
from neuralprophet import NeuralProphet, set_random_seed       
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
import datetime as dt                                         # https://docs.python.org/3/library/datetime.html

#-----------------------------------------------------------------------------------------------------------------------------------
# Hint: No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions
from Stock_TSA import Stock_Analysis 

#----------------------------------------------------------------------------------------------------------------------------------- 
# Prediction approaches
# 1) Neural Prophet: https://neuralprophet.com
# 2) TSA Forecast approach, aiming for yearly growth rate, combined with probabilistic model
#-----------------------------------------------------------------------------------------------------------------------------------

#General parameters
Stock_Name = ["^NDX"]   # NASDAQ 100 index taken from Yahoo Finance ticker
start="2018-06-09"      # start date, from which data is loaded till today
L = 0.7                 # Length scale as GP kernel parameter
N = 3                   # Number of GP optimization runs
split_factor = 0.5      # Train test split
option = 1              # User coice: 1 = real prediction , 2 = backtest




#Start program by downloading stock data 
warnings.filterwarnings("ignore")
set_random_seed(0)
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


#-----------------------------------------------------------------------------------------------------------------------------------
#Data basis for evaluation
data = Evaluation.DF
forecast_horizon=(1-split_factor)*(np.max(Time_idx)-np.min(Time_idx)) #according to Stock_TSA definition


#Fit neural prophet
nn_pr = NeuralProphet(seasonality_reg=0.5, seasonality_mode='multiplicative') 
nn_pr.fit(data, freq="D")  



# Predictions
future_DF = nn_pr.make_future_dataframe(data, periods = forecast_horizon.days, n_historic_predictions=len(data)) 
prophet_prediction = nn_pr.predict(future_DF)
nn_pr.plot(prophet_prediction)
plt.title('Neural Prophet model', fontsize=16)
plt.show()





#Visualization
plt.style.use("seaborn")
nn_pr.plot(prophet_prediction)
plt.plot(pd.DatetimeIndex(Time_training) ,forecast,'m-',linewidth=3, label = 'TSA Model Fit')
plt.fill(np.concatenate([pd.DatetimeIndex(Time_training), pd.DatetimeIndex(Time_training)[::-1]]),np.concatenate([forecast - 3 * sigma,(forecast + 3 * sigma)[::-1]]),
         alpha=.3, fc='y', ec='None', label='TSA: 99% confidence interval for training')
plt.plot(pd.DatetimeIndex(Time_test),forecast_future,'m-.',linewidth=3, label = 'TSA Forecast with Prediction Model')
plt.fill(np.concatenate([pd.DatetimeIndex(Time_test), pd.DatetimeIndex(Time_test)[::-1]]),np.concatenate([forecast_future - 3 * sigma_future,(forecast_future + 3 * sigma_future)[::-1]]),
         alpha=.2, fc='g', ec='None', label='TSA: 99% confidence interval for prediction')
plt.xlabel('Time', fontsize=16)
plt.ylabel('Closing Price', fontsize=16)
plt.legend()
plt.title(' "Neural Prophet" model (blue) and "Time Series Analysis" model (violet) ', fontsize=16)
plt.show()