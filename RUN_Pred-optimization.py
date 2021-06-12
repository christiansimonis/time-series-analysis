""" 2.3)  RUN_Pred-optimization: predictive portfolio optimization, utilizing the TSA model for prediction, followed by a monte carlo simulation. """
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

# For prediction, TSA models a yearly growth rate, combined with a probabilistic model.
# While the general growth rate of the stock or index is described in a domain model,
# especially non-efficient artifacts are modeled in a probabilistic way, 
# including these parts that the domain model is not capable of describing. 
# Assumptions rely on the course financial markets by Robert Shiller. 
# Information links (no promotion), see sources: 
# https://www.coursera.org/learn/financial-markets-global (no promotion)
# and https://en.wikipedia.org/wiki/Brownian_model_of_financial_markets
# To download relevant market data, yfinance library is used.
# For optimization, finquant library is used, see: https://finquant.readthedocs.io/en/latest/quickstart.html
# By combining market predictions with optimization algorithms a predictive portfolio optimization can be derived.
 
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
from finquant.portfolio import build_portfolio, EfficientFrontier  
import numpy as np                                                 
import yfinance as yfin                                            
import matplotlib.pyplot as plt                                    
from sklearn.preprocessing import MinMaxScaler                     
import pandas as pd                                                
import seaborn as sns                                              
import warnings                                                    # https://docs.python.org/3/library/warnings.html
import random                                                      # https://docs.python.org/3/library/random.html
import datetime as dt                                              # https://docs.python.org/3/library/datetime.html
#-----------------------------------------------------------------------------------------------------------------------------------
# Hint: No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions
from Stock_TSA import Stock_Analysis 

#----------------------------------------------------------------------------------------------------------------------------------- 
# Optimization of portfolio (1) based on based on past (2) based on predictions w/ option to add noise into forecast
#-----------------------------------------------------------------------------------------------------------------------------------


# General Parameters
Stock_Name      =         ["GOOG", "TCEHY", "DE", "MELI", "AMZN", "GC=F"]  # Google, Tencent, John Deere, Mercado Libre, Amazon, Gold
# Exemplary and arbitrary assumptions for interests rates. No responsibility is taken for correctness or completeness of historic, current or future data, models and / or predictions!
exp_interest    = np.array([1.07,  1.10,    1.10,  1.07,    1.08,  1.02 ])     # based on long-term expectation or analäö expected yearly interest can be defined
L_stock         = np.array([0.5,   0.2,     0.3,   0.9,     0.5,   1.2  ])     # based on sector, the Length scale as GP kernel parameter, defined for each stock



# Parameters for portfolio analysis 
start="2017-06-09"              # start date of time series analysis
end = "2021-06-12"              # end date of time series analysis
nr_mc = 5000                    # number of samples in MC simulation
risk_free_rate = 0.005          # risk free rate


# Predict future prices
add_noise = True                # Historic noise based on fit can be added to describe better expected vola
N = 5                           # Number of GP optimization runs
split_factor = 0.7              # Train test split
option = 1                      # User coice: 1 = real prediction , 2 = backtest

#-----------------------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
random.seed(0)
Portfolio = Stock_Analysis() #class call

#Download data    
stocks = yfin.download(Stock_Name, start= start, end = end)
stocks.columns = stocks.columns.to_flat_index()
stocks.columns = pd.MultiIndex.from_tuples(stocks.columns)
close = stocks.loc[:, "Close"].copy()
df_data = close.loc[:,Stock_Name] # Datafreame with closing values of relevant stocks
      
        
#Call optimization function for optimization based on history
historic_weights = Portfolio.optimize_pf(df_data, nr_mc, risk_free_rate)

#Bar plot of optimized weights
sns.set_theme(style="whitegrid")
ax = sns.barplot(data = historic_weights, palette="Blues_d")
plt.xlabel('Stocks', fontsize=16)
plt.ylabel('Weights of asset allocation', fontsize=16)
plt.title('Portfolio allocation based on historic optimization', fontsize=16)
plt.show()


#class call
Prediction = Stock_Analysis()  
Predictions_of_stocks = pd.DataFrame()

# loop over stocks
for i in range(len(Stock_Name)):
    
    #assign interest expectations
    Prediction.exp_interest = exp_interest[i] 
    

    #get data
    Prediction.end = end # to ensure consistant lenght
    Time, Stock,Time_idx  = Prediction.obtain_timeseries(Stock_Name[i],start)
   
    #conduct train test split
    Time_training, y_training, Time_test, y_test = Prediction.conduct_train_test_split(option,split_factor, Time,Stock)
    
    #Scale Time Space accordingly
    scaler = MinMaxScaler()
    scaler.fit(Time_training.reshape(-1,1)) # Fit on training data
    Time_training_scaled = scaler.transform(Time_training.reshape(-1,1))
    Time_test_scaled = scaler.transform(Time_test.reshape(-1,1))
    
    
    #Train on past data
    L = L_stock[i] #obtain defined parameter for a good model
    forecast, sigma, y_mdl = Prediction.fit_ForecastMdl(Time_training_scaled, y_training,L,N)
    residuum = y_training - forecast # calculate residuum
    noise = np.flip(residuum)
    
    #since optimization is based on the expected value, we want to represent the vola with the previous "non-efficient part", the TSA model could not explain well
    
    #Prediction
    forecast_future_wo_noise, sigma_future, y_mdl_future = Prediction.pred_ForecastMdl(Time_test_scaled)
    
    # Apply noise to represent vola
    if add_noise: #only if add_noise == True based on user parameter
        
        if len(forecast_future_wo_noise) < len(noise):
            forecast_future = forecast_future_wo_noise + noise[0:len(forecast_future_wo_noise)]
        else: #noise length shorter than prediction
            forecast_future = forecast_future_wo_noise.copy()
            forecast_future[:len(noise)] = forecast_future_wo_noise[:len(noise)].copy() + noise
                     

    else: # if add_noise == False
        forecast_future = forecast_future_wo_noise # no noise is added
        
    #Visualization
    Prediction.vis_results(Time_training,forecast,sigma,Time_test,forecast_future,sigma_future, Time,Stock, Stock_Name[i])
    plt.show() 

    #Store in data frame
    Predictions_of_stocks.loc[:,'Date'] = pd.to_datetime(Time_test)
    Predictions_of_stocks.loc[:,Stock_Name[i]] = forecast_future




#Predictive portfilio optimization based on forecast
Predictions_of_stocks.reset_index()
Predictive_Portfolio = Predictions_of_stocks.set_index(['Date']).copy()
predictive_weights = Portfolio.optimize_pf(Predictive_Portfolio, nr_mc, risk_free_rate)
plt.show



#Visualize normed prediction
normed_stocks = Predictive_Portfolio.copy()
normed_stocks.iloc[:,0:] = normed_stocks.iloc[:,0:]/(normed_stocks.iloc[0,0:])*100 #scaled t0 100 %
normed_stocks.plot()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Normalized predicted Closing Price in %', fontsize=16)
plt.legend()
plt.title(' Expected normed value of forecasting basis for portfolio optimization ', fontsize=16)
plt.show()


#Bar plot of optimized weights
sns.set_theme(style="whitegrid")
ax = sns.barplot(data = predictive_weights, palette="Blues_d")
plt.xlabel('Stocks', fontsize=16)
plt.ylabel('Weights of asset allocation', fontsize=16)
plt.title('Portfolio allocation based on predictive optimization', fontsize=16)
plt.show()

#Provide optimized weights
print('Comparison:')
print('_________________')
print('Historic_weights:')
print(historic_weights)
print('_________________')
print('Predictive_weights:')
print(predictive_weights)
print('_________________')

