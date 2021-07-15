#STEP1: ENV
import matplotlib.pyplot as plt

import numpy as np;  #print('numpy: %s' % np.__version__)
import pandas as pd; #print('pandas: %s' % pd.__version__)
from pandas import Series
from pandas.plotting import lag_plot, autocorrelation_plot

import matplotlib ; #print('matplotlib: %s' % matplotlib.__version__)
import matplotlib.pyplot as plt
import statsmodels; #print('statsmodels: %s' % statsmodels.__version__)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.stattools import adfuller

import sklearn; #print('sklearn: %s' % sklearn.__version__)
from sklearn.metrics import mean_squared_error
import scipy as sp; #print('scipy: %s' % sp.__version__)

from math import sqrt
import warnings 
warnings.filterwarnings("ignore")


#STEP2: Problem
def load_data(path):
    #from pandas import read_csv
    series = pd.read_csv(path, header=0, parse_dates=True, index_col=0, squeeze=True)
    print('Series length=%d' % len(series))
    print ('5 first lines of data:\n', series.head(5))
    return series


# STEP3: Test Harness
# 3.1 Validation Dataset: separate out a validation dataset
def split_set(series, split_percentage, filename1, filename2):
    split_point = int(len(series) * split_percentage)
    set1, set2 = series[0:split_point], series[split_point:]
    set1.to_csv(filename1 , header=False)
    set2.to_csv(filename2, header=False)
    return set1, set2

def prepare_data(series):
    return series.values.astype('float32')

# 3.2 Model Evaluation
# 3.2.1 Perf Measur
def model_evaluation_perf_measure(test, predictions):
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
    
    mape = np.mean(np.abs(predictions - test)/np.abs(test))  
    print('Mean Absolute Percentage Error (MAPE): %.3f' % mape)
    
    corr = np.corrcoef(predictions, test)[0,1]   # corr
    
    mins = np.amin(np.hstack([predictions[:,None], test[:,None]]), axis=1)
    maxs = np.amax(np.hstack([predictions[:,None],test[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    
    return rmse

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

#forecast_accuracy(fc, test.values)

# 3.2.2 Test Strategy
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

    # report performance
    rmse = model_evaluation_perf_measure(test, predictions)
    return rmse, predictions 


# Persistence-Baseline:
def baseline_predict(history):
    yhat = history[-1]
    return yhat

# ARIMA:
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

def arima_fitted_predict(dataset, model, bias, stationary=False, interval=1):    
    model_fit = ARIMAResults.load(model)
    bias = np.load(bias)
    output = float(model_fit.forecast()[0])
    if stationary :
        yhat = bias + inverse_difference(dataset, output, interval)
    else:
        yhat = bias + output
    print('Predicted: %.3f' % yhat)
    return yhat

def check_stattionary(data, epsilon=0.05):
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
    
def make_stationary(train,difference_intervals, stattionary_filename='stationary.csv'):
    stationary = Series(difference(train, difference_intervals))
    stationary.index = Series(train).index[difference_intervals:]
    # check if stationary
    result = check_stationary(stationary)
    if result :
        stationary.to_csv(stattionary_filename, header=False)
    else:
        print('It is still Not stationary. pls add more intervals!')
    stationary.plot()
    plt.show()

def hyperparam_tuning(train, test, p_values, d_values, q_values,
                          stationary=False, interval=1):
    best_score, best_cfg, best_predictions = float("inf"), None, list()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order=(p,d,q)
                try:
                    if stationary: 
                        rmse, predictions = model_evaluation(train, test, 
                                                             predict_model='ARIMA', 
                                                             order=order, 
                                                             stationary=stationary, 
                                                             interval=interval)
                        
                    else:
                        rmse, predictions = model_evaluation(train, test,
                                                             predict_model='ARIMA', 
                                                             order=order)
                        


                    print('Arima Order(%d,%d,%d) RMSE: %.3f' % (p,d,q,rmse))
                    if rmse < best_score:
                        best_score, best_cfg, best_predictions = rmse, order, predictions
                except:
                    continue
    print ('Best ARIMA%s Rms=%.3f' % (best_cfg, best_score))
    return best_score, best_cfg, best_predictions

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
    print(model_fit.summary())
    model_fit.plot_predict(dynamic=False)
    plt.show()
    model.plot_diagnostics(figsize=(7,5))
    plt.show()

    # save model
    model_fit.save(model_filename)
    np.save(model_bias_filename, [bias])
    return

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
    print (type(y))
    print (len(y), len(validation), len(X), len(data), len(history))
    y_temp = y[1:]
    rmse, predictions = model_evaluation(history, y_temp, predict_model,order, stationary,
                                         interval,np.load('model_bias.npy'))

    final_predictions.extend(predictions)
    print (type(y))
    plt.plot(y.values)
    plt.plot(final_predictions, color='red')
    plt.show()
    

# STEP 5: Data Analysis Plots
def data_analysis_plot_simple(series):
    #5.1: summary Statics
    print ('5.1: Series Description is:\n', series.describe())
    
    #5.2: Line Plot
    import matplotlib.pyplot as plt 
    series.plot(style='k-')
    plt.show()
    series.plot(style='k.')
    plt.show()
    
    #5.4: Density Plot  
    plt.figure(1)
    plt.subplot(211)
    series.hist()
    plt.subplot(212)
    series.plot(kind='kde')
    plt.show()
       
def data_analysis_plot_seasonal_box(series, start, end, freq='A'):
    #5.3: Seasonal Line Plot     
    groups = series[start:end].groupby(pd.Grouper(freq=freq))
    years = pd.DataFrame()    
    for name, group in groups:
        years[name.year] = group.values
    #years.plot(subplots=True, legend = False)
    years.plot()
    plt.show()

    #5.5:  Box & Wishker Plot   
    years.boxplot()
    plt.show()

def data_analysis_plot_lag_correlation(series, lag=1):
    lag_plot(series, lag)
    plt.show() 

    # Pearson Correlation Coefficients
    autocorrelation_plot(series)
    plt.show()
    
    #plot_acf(series, lags=lag, ax=plt.gca())
    plot_acf(series, ax=plt.gca())
    plt.show()
    
    #plot_pacf(series, lags=lag, ax=plt.gca())
    plot_pacf(series, ax=plt.gca())
    plt.show()

def plot_sets_results(train, test, predictions):
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()

    plt.plot(train)
    plt.plot([None for i in train] + [x for x in test])
    plt.plot([None for i in train] + [x for x in predictions])
    plt.show()
    
    
def plot_Residuals( test, predictions):
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    plt.figure()
    plt.subplot(221)
    residuals.hist(ax=plt.gca())
    plt.subplot(222)
    residuals.plot(kind='kde', ax=plt.gca())
    plt.subplot(223)
    plot_acf(residuals, ax=plt.gca())
    plt.subplot(224)
    plot_pacf(residuals, ax=plt.gca())
    plt.show()
    
# Step1: Load And Analyse data     
path = "/kaggle/input/champagneproject/monthly_champagne_sales.csv"
series = load_data(path)
#data_analysis_plot_simple(series)
#data_analysis_plot_seasonal_box(series,'1964', '1970', freq='A')
#data_analysis_plot_lag_correlation(series,1)

#Step2: Create train, cv, test sets
data, validation = split_set(series, 0.89, 'dataset.csv', 'validation.csv')
print('Dataset %d, Validation %d' % (len(data), len(validation)))
train, test = split_set(data, 0.50, 'train.csv', 'test.csv')
print('Train %d, Test %d' % (len(train), len(test)))
train = prepare_data(train)
test = prepare_data(test)
    
#Step3: Baseline/Naive
#rmse, predictions  = model_evaluation(train, test, predict_model='baseline')
#plot_sets_results(train, test, predictions)

#Step5: First Arima
#rmse, predictions = model_evaluation(train, test, predict_model='ARIMA', order=(1,0,0))
#plot_sets_results(train, test, predictions)

#Step6: Arima using Manual config based on plots
months_in_year = 12
#make_stationary(train,difference_intervals=months_in_year)
#rmse, predictions = model_evaluation(train, test, predict_model='ARIMA', order=(0,0,1), 
#                 stationary=True, interval=months_in_year)
#plot_sets_results(train, test, predictions)

#Step7: Hyper-param Tuning
#p_values = range(0,7) #[0,1,2,4,6,8,10]
#d_values = range(0,3)
#q_values = range(0,7)
#best_score, best_cfg, predictions = hyperparam_tuning(train, test, 
#                                                      p_values, d_values, q_values,
#                                                      stationary=True, 
#                                                      interval=months_in_year)
#plot_sets_results(train, test, predictions)

# Find  bias
#rmse, predictions = model_evaluation(train, test, predict_model='ARIMA', order=(0,0,1), 
#                                     stationary=True, interval=months_in_year)
#plot_sets_results(train, test, predictions)
#plot_Residuals(test, predictions)
#bias = 165

#Step7: Finaliz model using best found hyperparam
#! rm /kaggle/working/model*
#finalize_model(data, months_in_year, (0,0,1), bias,'model.pkl','model_bias.npy')
# ! ls /kaggle/working

#Step8: Make prediction
#prdct = arima_fitted_predict(data, 'model.pkl', 'model_bias.npy',True, months_in_year)

#Step9: Validate Model
#validate_model(data, validation, 'ARIMA',(0,0,1), True, months_in_year,
#               'model.pkl','model_bias.npy')