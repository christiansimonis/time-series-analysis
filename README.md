# time-series-analysis (TSA)


There have been several attempts to predict financial markets and stock prices using time series analysis. Many of them were not successful!
Neither trading nor investing decisions should be influenced by this repository or the code, which is built only to introduce and demonstrate a methodology for time series modeling.
No responsibility is taken for correctness or completeness of historic, current or future data, models, information and / or predictions!
https://github.com/christiansimonis/time-series-analysis

![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/forecast_exmpl.JPG)
![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/pred_dailyret.JPG)



# 1) Stock_TSA class

Approach: The TSA class models a yearly return rate, combined with a probabilistic model.
While the general growth rate of the stock or index is described in a domain model, especially non-efficient artifacts are modeled in a probabilistic way, including these parts that the domain model is not capable of describing. Assumptions rely on the course _financial markets_ by Robert Shiller. 
Information links (no promotion): https://www.coursera.org/learn/financial-markets-global and https://en.wikipedia.org/wiki/Brownian_model_of_financial_markets

![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/mdl_perf.JPG)

 
# 2) RUN files demonstrate use cases, such as:

* 2.1) RUN_Stock_historic_TSA: evaluates the performance of the Stock_TSA class based on historical data, using the Nasdaq 100 index as an example.

![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/res_eval.JPG) 
![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/pred_eval.JPG)

* 2.2) RUN_Stock-Forecast: demonstrates two forecasting methodologies, using the class Stock_TSA and Neural Prophet.

![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/pred_future.JPG)

* 2.3) RUN_Pred-optimization: enables a predictive portfolio optimization, utilizing the class Stock_TSA for prediction, followed by a monte carlo simulation.

![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/pred_opt.JPG)
![alt text](https://github.com/christiansimonis/time-series-analysis/blob/master/vis/pred_weights.JPG)
             
           
            


# Disclaimer
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Acknowledgements
* https://www.coursera.org/learn/financial-markets-global (no promotion)
* https://finquant.readthedocs.io
* https://pypi.org/project/yfinance/

Thanks and reference to:
(Name,                                 Version,                       License)  
* FinQuant,                            v0.2.2,                        MIT License,                                 Copyright (c) 2019, Frank Milthaler.             
* numpy,                               v1.19.5,                       BSD,                                         Copyright (c) 2005-2020, NumPy Developers.   
* yfinance,                            v0.1.59,                       Apache License, Version 2.0,                 Copyright (c) January 2004, Ran Aroussi.
* matplotlib,                          v3.4.2,                        Python Software Foundation License,          Copyright (c) 2002 - 2012, John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team.
* scikit-learn,                        v0.23.1,                       BSD 3-Clause License,                        Copyright (c) 2007-2021, The scikit-learn developers.
* pandas,                              v1.2.4,                        BSD 3-Clause License                         Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team.
* seaborn,                             v0.11.1,                       BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2012-2021, Michael L. Waskom.
* scipy,                               v1.5.2,                        BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2001-2002, Enthought, Inc.  2003-2019, SciPy Developers.
* neuralprophet,                       v0.2.7,                        MIT License,                                 Copyright (c) 2020, Oskar Triebe.

# Contact and collaboration
* [LinkedIn](https://www.linkedin.com/in/christiansimonis/)
* [GitHub](https://github.com/login?return_to=%2Fchristiansimonis)
* [Forks](https://github.com/christiansimonis/time-series-analysis/network/members)
* [Stargazers](https://github.com/christiansimonis/time-series-analysis/stargazers)
