# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:15:02 2020
Ref: https://medium.com/datadriveninvestor/time-series-prediction-using-sarimax-a6604f258c56
https://github.com/arshren/TimeSeries/blob/master/Stock%20Price%20APPL.ipynb
@author: Mali
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%matplotlib inline

dataset= pd.read_csv('datasets/AAPL.csv')
dataset.head(2)
dataset.info()

# data prepare
dataset['Mean'] = (dataset['Low'] + dataset['High'])/2
dataset.head(2)

# prediction based on X steps previous
steps=-1
dataset_for_prediction= dataset.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift(steps)
dataset_for_prediction.head(3)

# Dropping columns with null values
dataset_for_prediction=dataset_for_prediction.dropna()

# Creating Date as the index of the DataFrame
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# Plot the mean stock prices for the current day 
dataset_for_prediction['Mean'].plot(color='green', figsize=(15,2))
plt.legend(['Next day value', 'Mean'])
plt.title('Tyson Opening Stock Value')

# Plotting volume of Apple stocks sold daily
dataset_for_prediction['Volume'].plot(color='blue', figsize=(15,2))
plt.title('Apple Stock Volume')

# Normalizing the input and target features
"""
Since the stock prices and volume are on different scale, 
we need to normalize the data. 
We use MinMaxScaler, it will scale the data to a fixed range between 0 to 1
Scaling the input features- Low, High, Open, Close, Volume, Adjusted Close and Mean
"""
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[
    ['Low', 'High','Open', 'Close', 'Volume', 'Adj Close', 'Mean']])
scaled_input =pd.DataFrame(scaled_input)
X= scaled_input

"""
Scaling the output features - Actual. 
We are using a different instance of MinMaxScaler here. 
This will allow us to perform inverse transform the predicted stock prices later easily.
"""
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output)
y=scaler_output

X.rename(columns={0:'Low', 1:'High', 2:'Open', 
                  3:'Close', 4:'Volume', 5:'Adj Close', 
                  6:'Mean'}, inplace=True)
X.head(2)

y.rename(columns={0:'Stock Price next day'}, inplace= True)
y.index=dataset_for_prediction.index
y.head(2)


# Splitting the data into training and test set
# training set will be 70% and test set will be 30% of the entire data set
train_size=int(len(dataset) *0.7)
test_size = int(len(dataset)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()



# Understanding the Time series data
import statsmodels.api as sm
seas_d=sm.tsa.seasonal_decompose(X['Mean'],model='add',freq=365);
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()


# Check Stationary:
from statsmodels.tsa.stattools import adfuller
def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)


y_test=y['Stock Price next day'][:train_size].dropna()
test_adf(y_test, " Stock Price")
test_adf(y_test.diff(), 'Stock Price') # i diff for making stationary

# Building the Model
from pmdarima.arima import auto_arima
#step_wise=auto_arima(train_y, 
#                     exogenous= train_X,
#                     start_p=1, start_q=1, 
#                     max_p=7, max_q=7, 
#                     d=1, max_d=7,
#                     trace=True, 
#                     error_action='ignore', 
#                     suppress_warnings=True, 
#                     stepwise=True)
#step_wise.summary()


# Auto_arima suggests a SARIMAX with order=(0,1,1). So we use it:
from statsmodels.tsa.statespace.sarimax import SARIMAX
model= SARIMAX(train_y, exog=train_X, order=(0,1,1),
               enforce_invertibility=False, enforce_stationarity=False)
results= model.fit()
predictions= results.predict(start =train_size, 
                             end=train_size+test_size+(steps)-1,
                             exog=test_X)

forecast_1= results.forecast(steps=test_size-1, exog=test_X)
#Steps is an integer value that specifies the number of steps to 
#forecast from the end of the sample.


# Plot the predictions
act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Stock Price next day']
predictions.rename(columns={0:'Pred'}, inplace=True)

predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')
predictions['Pred'].plot(legend=True, color='red', figsize=(20,8))


forecast_apple= pd.DataFrame(forecast_1)
forecast_apple.reset_index(drop=True, inplace=True)
forecast_apple.index=test_X.index
forecast_apple['Actual'] =scaler_output.iloc[train_size:, 0]
forecast_apple.rename(columns={0:'Forecast'}, inplace=True)
forecast_apple['Forecast'].plot(legend=True)
forecast_apple['Actual'].plot(legend=True)


# Evaluating the Model
from statsmodels.tools.eval_measures import rmse
error=rmse(predictions['Pred'], predictions['Actual'])
print (error)


# Scaling back to original values
trainPredict = sc_out.inverse_transform(predictions[['Pred']])
testPredict = sc_out.inverse_transform(predictions[['Actual']])










