# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:32:58 2020

@author: Mali
Ref: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
"""

import pandas as pd
#from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
#path = 'https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv'
path = "datasets/monthly_champagne_sales.csv"
#df = pd.read_csv(path, names=['value'], header=0)
#df = pd.read_csv(path, header=0, parse_dates=True, squeeze=True)
df = pd.read_csv(path, header=0, parse_dates=True, index_col=0, squeeze=True)
print('Series length=%d' % len(df))
print ('5 first lines of data:\n', df.head(2))

model = pm.auto_arima(df.values, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)


print(model.summary())
model.plot_diagnostics(figsize=(7,5))

pred = model.predict_in_sample(start=93, end=105)
print (pred)

forecast1 = model.predict(n_periods=10)
print (type(forecast1))
print(sum(forecast1))
#forecast2 = model.predict(n_periods=3)
#forecast3 = model.predict(n_periods=4)
#print(forecast1,forecast2, forecast3 )



"""
Explaining the parameters for auto_arima
exogenous : An optional 2-d array of exogenous variables. If provided, these variables are used as additional features in the regression operation. If an ARIMA is fit on exogenous features, it must be provided exogenous features for making predictions.
seasonal : Whether to fit a seasonal ARIMA. By default it is set to True. If we enable the seasonal parameter then we need to provide the P,D and Q parameters.
m: Refers to the period for seasonal differencing, number of periods in each season.
d : auto_arima works by conducting differencing tests and this is parameter used for determining the order of differencing
start_p, max_p, start_q, max_q : We fit the model based on these defined ranges
trend : Trend of the time series. “c” for constant trend and “t” for linear trend and when we have both we can specify “ct”
error_action : default behavior is to “warn”, In our case we are ignoring the error
trace : will print status on the fits
stepwise : specifies if we want to use stepwise algorithm. stepwise algorithm is significantly faster than fitting all hyper-parameter combinations and is less likely to over-fit the model.
"""