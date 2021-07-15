# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:39:15 2020
@author: Mali
Ref: https://towardsdatascience.com/how-to-forecast-sales-with-python-using-sarima-model-ba600992fa7d

"""
import itertools 
import statsmodels.api as sm 
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'

def hyperparamTunning_sarimax(series, pdq, seasonal_pdq ):
    best_cfg = {"pdq":(0,0,0),"seasonal_pdq":(0,0,0,12)}
    best_aic =  float("inf")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(series,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = model.fit()
                print('ARIMA{}x{} - AIC:{}'.format(param,param_seasonal,results.aic))
                if results.aic < best_aic :
                          best_aic = results.aic
                          best_cfg ['pdq'] = param
                          best_cfg['seasonal_pdq'] = param_seasonal
            except: 
                continue
    return best_cfg, best_aic
            
# Load Data
path = "datasets/monthly_champagne_sales.csv"
series = pd.read_csv(path, header=0, parse_dates=True, index_col=0, squeeze=True)
print('Series length=%d' % len(series))
print ('5 first lines of data:\n', series.head(2))

# Hyper param Tunning
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#best_cfg, best_aic = hyperparamTunning_sarimax(series, pdq, seasonal_pdq)    
best_cfg = {'pdq': (0, 1, 1), 'seasonal_pdq':(1, 1, 1, 12)}
best_aic = float(1259.9425546033115)
print('The Best SARIMA{}x{} - AIC:{}'.format(best_cfg['pdq'], best_cfg['seasonal_pdq'],
                                     best_aic))

# Running the best model
model = sm.tsa.statespace.SARIMAX(series,
                                order=best_cfg['pdq'],
                                seasonal_order=best_cfg['seasonal_pdq'],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = model.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))
plt.show()


# Prediction: CV Test
pred = results.get_prediction(start=pd.to_datetime('1968-01-01'), dynamic=False)
#print(type(pred)) #PredictionResultsWrapper
ax = series['1964':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', 
                         alpha=.7, figsize=(14, 4))
pred_ci = pred.conf_int()
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('champagne_sold')
plt.legend()
plt.show()


# RMSE, RMS, & MAPE
forecast_cv = pred.predicted_mean
actual_cv = series['1968-01-01':]
mape = np.mean(np.abs(forecast_cv - actual_cv)/np.abs(actual_cv))  # MAPE
mse = ((forecast_cv - actual_cv) ** 2).mean()
print('The MAPE is {}'.format(round(mape, 2)))
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

forecast_cv = pred.predicted_mean
print('\n\n\n forecast_cv.head(12):\n',forecast_cv.head(12))
print('\n\n\n actual_cv.head(12):\n', actual_cv.head(12))



# Forcast next 12 month
pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = series.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

print('\n\n\n pred_ci.head(24):\n', pred_ci.head(24))
forecast = pred_uc.predicted_mean
print('\n\n\n forecast.head(12)', forecast.head(12))

fc = forecast.values[-12:]
print('\n\n\n forecast.tail(12)', fc)
pr = forecast_cv.values[-12:]
ac = actual_cv.values[-12:]
print('\n\n\n forecast_cv.tail(12):\n',pr)
print('\n\n\n actual_cv.tail(12):\n', ac)

