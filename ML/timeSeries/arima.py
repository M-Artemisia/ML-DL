# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:26:18 2020

@author: Mali
"""

#*************************STEP0: ENV *************************
import matplotlib.pyplot as plt

import numpy as np;  #print('numpy: %s' % np.__version__)
import pandas as pd; #print('pandas: %s' % pd.__version__)
from pandas import Series
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
#import scipy as sp; #print('scipy: %s' % sp.__version__)

from math import sqrt
import warnings 
warnings.filterwarnings("ignore")


#*************************Utils*************************
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff) 

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

#*************************Plots*************************

def plot_simple(series):
    #5.1: summary Statics
    print ('5.1: Series Description is:\n', series.describe())
    
    import matplotlib.pyplot as plt
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Line Plot')
    series.plot(style='k-')
    plt.show()
    series.plot(style='k.')
    plt.show()
    
    #5.4: Density Plot  
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Histogram Plot')
    series.hist()
    plt.subplot(212)
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Gaussian Plot')
    series.plot(kind='kde')
    plt.show()
       
def plot_seasonal_box(series, start, end, freq='A'):
    #5.3: Seasonal Line Plot     
    groups = series[start:end].groupby(pd.Grouper(freq=freq))
    years = pd.DataFrame()    
    for name, group in groups:
        years[name.year] = group.values
    #years.plot(subplots=True, legend = False)
    years.plot()
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Seasonal Plot')
    plt.show()

    #5.5:  Box & Wishker Plot 
    plt.title('Canndle Plot')
    years.boxplot()
    plt.show()

def plot_sesonalDecompose(series):
    result_add = sm.tsa.seasonal_decompose(series, model='additive',
                                              extrapolate_trend='freq')   
    result_mul = sm.tsa.seasonal_decompose(series, model='multiplicative',
                                              extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (10,10)})
    plt.title('Seasonal Decompose Plot')
    plt.xlabel('Time')
    plt.ylabel('sale')
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.show()
    
def plot_lag(series, lag=1):
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Lag Scatter')
    lag_plot(series, lag)
    plt.show() 

def plot_correlation(series, lag=1): 
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.subplot(211)
    plt.title('ACF: Pearson Correlation Coefficients')
    plot_acf(series, ax=plt.gca())
    
    plt.subplot(212)
    #plot_pacf(series, lags=lag, ax=plt.gca())
    plt.title('Partial ACF')
    plot_pacf(series, ax=plt.gca())    
    plt.show()

def plot_sets_results(train, test, predictions):
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Test & Prediction Set')
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()

    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Train, Test & Prediction Set')
    plt.plot(train)
    plt.plot([None for i in train] + [x for x in test],label='original')
    plt.plot([None for i in train] + [x for x in predictions], label='predicted')
    plt.show()
    
def plot_Residuals(test, predictions):
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('Residual Graph (between Test & Prediction)')
    plt.subplot(221)
    residuals.hist(ax=plt.gca())
    plt.subplot(222)
    residuals.plot(kind='kde', ax=plt.gca())
    plt.subplot(223)
    plot_acf(residuals, ax=plt.gca())
    plt.subplot(224)
    plot_pacf(residuals, ax=plt.gca())
    plt.show()
    
#**********************STEP2: Data, Sets*************************
def load_data(path):
    #from pandas import read_csv
    series = pd.read_csv(path, header=0, parse_dates=True, index_col=0, squeeze=True)
    print('Series length=%d' % len(series))
    #print('Series info:', series.info)
    print ('2 first and 2 last lines of data:\n', series.head(2), series.tail(2))
    return series

def split_set(series, split_percentage, filename1, filename2):
    split_point = int(len(series) * split_percentage)
    set1, set2 = series[0:split_point], series[split_point:]
    set1.to_csv(filename1 , header=False)
    set2.to_csv(filename2, header=False)
    return set1, set2

def split_set2(series, train_prcnt, test_prcnt):
    size = int(len(series))
    train_index = int(size*train_prcnt)
    test_index = train_index + int(size*test_prcnt)
    train= series[0:train_index]
    test = series[train_index:test_index] 
    cv = series[test_index:]
        
    train.to_csv('train.csv' , header=False)
    test.to_csv('test.csv', header=False)
    cv.to_csv('validation.csv', header=False)
    return train, test, validation

def prepare_data(series):
    return series.values.astype('float32')

# STEP3: Accuracy Check
def model_evaluation_perf_measure(test, predictions):
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)  
    return rmse

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = 1 #np.corrcoef(forecast, actual)[0,1]   # corr
    #mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    #maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    #minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = 1 #acf(forecast-actual)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr})

#forecast_accuracy(fc, test.values)

#*************************STEP3: Data Preparation*************************
def check_stationary(data, epsilon=0.05):      
    rolmean = pd.Series(data).rolling(window=12).mean()
    rolstd = pd.Series(data).rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(12, 8))
    orig = plt.plot(data, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    
    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
    if result[1] < epsilon :
        print('data is stationary')
        return True
    else:
        print('data is NOT stationary')
        return False
    
def make_stationary(train,difference_intervals):
    #Also u can use: https://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
    stationary = Series(difference(train, difference_intervals))
    stationary.index = Series(train).index[difference_intervals:]
    if not check_stationary(stationary):
        print('It is still Not stationary. pls add more intervals!')
    plot_correlation(stationary)
    stationary.plot()
    plt.show()
    return

def make_stationary_method(data, intervals, method="diff", fname='stationary.csv'):
    if method=='MA': 
        moving_avg = pd.Series(data).rolling(interval).mean()
        stat_data = data - moving_avg
        stat_data.dropna(inplace=True) # first 6 is nan value due to window size            
    else:
        stat_data = data - data.shift()

    if check_stationary(stat_data):
        stationary.to_csv(fname, header=False)
    else:
        print('It is still Not stationary!')
    
    plot_correlation(stat_data )
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('The made stationary data')
    stat_data.plot()
    plt.show()
    return stat_data 
    

# *************************STEP4: Model Training*************************
# Persistence, Baseline, Naive:
def baseline_predict(history):
    yhat = history[-1]
    return yhat

def model_evaluation(train, test, predict_model='baseline', order=(1,0,0), 
                     stationary=False, interval=1, bias=0):
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        if predict_model == 'baseline':
            yhat = baseline_predict(history)
            #yhat = history[-1]
        elif predict_model == 'ARIMA' or predict_model=='arima' or predict_model == 'Arima':
            if stationary : 
                yhat = arima_predict(history, order, stationary, interval, bias)
            else:
                yhat = arima_predict(history, order)
        predictions.append(yhat)
        obs = test[i]                   # observation
        history.append(obs)  
        #print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    acc = forecast_accuracy(predictions, test )
    return acc, predictions 

# ARIMA:
def arima_predict(history, arima_order, stationary=False, interval=1, bias=0):    
    if stationary :
        model = ARIMA(difference(history, interval), arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        output = model_fit.forecast()
        yhat = output[0] 
        yhat = bias + inverse_difference(history, yhat, interval)
    else:
        model = ARIMA(history, arima_order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = bias + output[0]
    return yhat

def fitted_predict(dataset, order=(1,0,0), interval=1, bias=0,step=1):
    # walk-forward validation
    history = [x for x in dataset]
    predictions = list()
    for i in range(step):
        #model_fit = ARIMAResults.load(model)
        #bias = np.load(bias)
        #output = model_fit.forecast()
        model = ARIMA(difference(history, interval), order)
        model_fit = model.fit(trend='nc', disp=0)
        output = model_fit.forecast()
        
        yhat = output[0] 
        yhat = bias + inverse_difference(history, yhat, interval)
        predictions.append(yhat)
        history.append(yhat)  
    return predictions 
    



def arima_fitted_predict(dataset, model, bias, stationary=False, interval=1, step=1):    
    model_fit = ARIMAResults.load(model)
    bias = np.load(bias)
    output = model_fit.forecast()
    print('output type is:',type(output))
    print('output is:',output)
    output = output[0]
    print('output[0] type is:',type(output))
    print('output[0] is:',output)
    yhat=list()
    if stationary and step>1:
        for prd in output:
            print('prd in output is: ',prd)
            yhat.append(bias+inverse_difference(dataset, float(prd),interval))
    elif stationary and step==1:
        yhat.append(bias + 
                    inverse_difference(dataset, float(output),interval))
    else:
        yhat.append(bias + output)
    print('yhat type is: ', type(yhat))
    return yhat
       
def new_predict(data, start_index, end_index, model):
    model_fit = ARIMAResults.load(model)    
    my_forecast = model_fit.predict(start=start_index, end=end_index)
    print('in new predict type is', type(data))
    print(', my_forecast type is', type(my_forecast))
    # visualization
    plt.figure(figsize=(22,10))
    plt.plot(data.values[start_index:end_index], label = "original")
    plt.plot(my_forecast+12,label = "predicted")
    plt.title("Time Series Forecast-New Predict")
    plt.xlabel("Date")
    plt.ylabel("Sale")
    plt.legend()
    plt.show()

# predict all path
# fit model
def new_forcecast(data, model):
    model_fit = ARIMAResults.load(model)
    my_forecast = model_fit.predict()
    #error = mean_squared_error(data, my_forecast)
    #print("error: " ,error)
    # visualization
    plt.figure(figsize=(22,10))
    plt.plot(data.values,label = "original")
    plt.plot(my_forecast,label = "predicted")
    plt.title("Time Series Forecast-NEW Forcast")
    plt.xlabel("Date")
    plt.ylabel("Sale")
    plt.legend()
    plt.savefig('graph.png')
    plt.show()
#*************************STEP5: Hyper-parameter Tuning*************************
def hyperparam_tuning(train, test, p_values, d_values, q_values,
                          stationary=False, interval=1):
    best_score, best_cfg, best_predictions = float("inf"), None, list()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order=(p,d,q)
                try:
                    if stationary: 
                        acc, predictions = model_evaluation(train, test, 
                                                             predict_model='ARIMA', 
                                                             order=order, 
                                                             stationary=stationary, 
                                                             interval=interval)
                        
                    else:
                        acc, predictions = model_evaluation(train, test,
                                                             predict_model='ARIMA', 
                                                             order=order)
                        


                    print('Arima Order(%d,%d,%d) MAPE: %.3f' % (p,d,q,acc['mape']))
                    if acc['mape'] < best_score:
                        best_score, best_cfg, best_predictions = acc['mape'], order, predictions
                except:
                    continue
    print ('Best ARIMA%s Rms=%.3f' % (best_cfg, best_score))
    return best_score, best_cfg, best_predictions


#*************************STEP6: Cross Validation Check*************************

def validate_model(data, validation,
                   predict_model='ARIMA',order=(0,0,1), stationary=True, interval=1,
                   model='model.pkl', bias='model_bias.npy'):
    X = data 
    y = validation
    yhat = arima_fitted_predict(data, model, bias, stationary, interval)
    final_predictions = list()
    final_predictions.append(yhat)
    #print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

    history = [x for x in X]
    history.append(y[0])
    #print (len(y), len(validation), len(X), len(data), len(history))
    y_temp = y[1:]
    acc, predictions = model_evaluation(history, y_temp.values, predict_model,order, stationary,
                                         interval,np.load('model_bias.npy'))

    final_predictions.extend(predictions)
    plt.plot(y.values,label='Validation-Actual')
    plt.plot(final_predictions, color='red', label='Validatin-Prediction')
    plt.show()
    return predictions
    
#************************* STEP7: Save Model*************************
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
               
def finalize_model(data, interval=1,arima_order=(1,0,0), bias=0, 
                   model_filename='model.pkl', model_bias_filename='model_bias.npy'):
    from scipy.stats import boxcox
    ARIMA.__getnewargs__ = __getnewargs__
    
    diff = difference(data, interval)
    # fit model
    model = ARIMA(diff, order=arima_order)
    model_fit = model.fit(trend='nc', disp=0)
    
    # Actual vs Fitted
    print('fitted model summary:\n\n', model_fit.summary())
    #model_fit.plot_predict(dynamic=False)
    #plt.title('model_fit.plot_predict')
    plt.show()

    # save model
    model_fit.save(model_filename)
    np.save(model_bias_filename, [bias])
    return



#***************************************************************************
#                                                                          *
#                               Main Body                                  *
#                                                                          *
#***************************************************************************

# Step1: Load And Analyse data     
#path = "datasets/monthly_champagne_sales.csv"
#path = "datasets/filtered_sample_output.csv"
path = "datasets/filtered_train_data.csv"
data = pd.read_csv(path)
print(data.info())
series = load_data(path)
#plot_simple(series)
#plot_sesonalDecompose(series)
#plot_lag_correlation(series,1)
#plot_seasonal_box(series,'1964', '1970', freq='A')

"""
#Step2: Create train, cv, test sets
data, validation = split_set(series, 0.89, 'dataset.csv', 'validation.csv')
print('Dataset %d, Validation %d' % (len(data), len(validation)))
train, test = split_set(data, 0.70, 'train.csv', 'test.csv')
print('Train %d, Test %d' % (len(train), len(test)))
train = prepare_data(train)
test = prepare_data(test)
    
#Step3: Baseline/Naive
#acc, predictions  = model_evaluation(train, test, predict_model='baseline')
#plot_sets_results(train, test, predictions)


#Step4: remove Seasonal correlation by making stationary
interval = 1
Stattionary = False

if not check_stationary(train): 
    months_in_year = 12
    interval = months_in_year
    stationary = True 
    make_stationary(train, interval)
acc, predictions = model_evaluation(train, test, 'ARIMA', (0,0,1), stationary, interval)
plot_sets_results(train, test, predictions)

#Step7: Hyper-param Tuning
p_values = range(0,7) #[0,1,2,4,6,8,10]
d_values = range(0,3)
q_values = range(0,7)
#best_score, best_cfg, predictions = hyperparam_tuning(train, test, 
#                                                      p_values, d_values, q_values,
#                                                      stationary, interval)
#plot_sets_results(train, test, predictions)
best_cfg = (0,0,1) #= (1,0,0)
# Find  bias 
#acc, predictions = model_evaluation(train, test, 'ARIMA', best_cfg, stationary, interval)
#plot_sets_results(train, test, predictions)
#plot_Residuals(test, predictions)

bias = 165.90

#Step7: Finaliz model using best found hyperparam and bias
finalize_model(data, interval, best_cfg, bias,'model.pkl','model_bias.npy')

#Step8: Make prediction
# New predict
#print('\n\n\n befor new predict\n\n\n')
#new_predict(data, 75, 93, 'model.pkl')
#new_forcecast(data, 'model.pkl')

step = 20
#fc = arima_fitted_predict(data, 'model.pkl', 'model_bias.npy', 
#                            stationary, interval, step)
fc = fitted_predict( data, (1,0,0), interval, bias, step)
# visualization
plt.plot(data.values, label = "original")
plt.plot([None for i in data.values]+ [x for x in fc],label = "Forecast")
plt.title('Time Series Forecast. Fitted_Predict_Func. %d steps' % step)
plt.xlabel("Date")
plt.ylabel("Sale")
plt.show()
    
#fc = [i[0] for i in fc]

#Step9: Validate Model
pred = validate_model(data, validation, 'ARIMA',(0,0,1), stationary, interval,
              'model.pkl','model_bias.npy')
pred = [i[0] for i in pred]
#plot_sets_results(data, validation, pred)




# Compare Models
pred_auto = [6733.77767084, 4930.35541495, 7926.53191397, 7812.94233582, 
               2550.45051427, 5421.6552144, 4230.34209351, 5129.37868533, 
               4444.00892039, 5328.33630339, 4111.30013996, 3041.95228849]
fc_auto = [6733.77767084, 4930.35541495, 7926.53191397,
                    7812.94233582, 2550.45051427, 5421.6552144,
                    4230.34209351, 5129.37868533, 4444.00892039,
                    5328.33630339, 4111.30013996, 3041.95228849,
                    6589.43545729]

act = series.values[93:]
act2 =[ 6981, 9851,12670, 4348, 3564, 4577, 4788, 4618, 5312, 4298, 1413, 5877]

fc_sarima = [ 7071.03120902, 10054.70343898, 12691.08716509,  4518.02660905,
             3761.72427995, 4653.13032398, 4857.73743419, 4819.41361089,
             5316.37947776, 4392.26214771, 1804.43852346, 5833.49573401]
pred_sarima = [ 6811.69452924, 10202.65037657, 13062.57006987,  4310.68302403,
                3570.64520314, 4523.06238243, 4855.77247604, 5162.86874837,
                5026.85244167, 4664.47332493, 2012.73699046, 5816.30115795]

#data = np.array([cv_auto_prd, cv_auto_forecast, forecasted])


data_f = list(zip(pred, pred_auto, pred_sarima, act2, act,
                  fc_auto,fc_sarima ,fc))
cols = ['prediction', 'pred_auto', 'pred_sarima', 'actual2', 'actual',
                  'fc_auto','fc_sarima' ,'forecast']
df_compare_models = pd.DataFrame(data_f, columns=cols)
print(df_compare_models)

acc = forecast_accuracy(pred, act[1:])
acc2 = forecast_accuracy(pred_auto, act)
acc3 = forecast_accuracy(pred_sarima, act)
print(acc,'\n',acc2,'\n',acc3)
"""