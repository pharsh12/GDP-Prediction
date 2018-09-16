# -*- coding: utf-8 -*-
"""
Created on Sun May 13 14:21:09 2018

@authors: Harsh Nisar, Ketul Patel, Harsh Patel and Shrinivas Phatale
"""
#############################################
#Import Libraries
#############################################

import matplotlib.pyplot as plt
import quandl
import numpy as np
import pandas as pd
from sklearn import linear_model
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot

##########################################################
#API
###########################################################

quandl.ApiConfig.api_key = "eWQyiDgtMtFJ6YgJkAkB"

###########################################################
#DataFrame
###########################################################

df = quandl.get('BEA/T10105_Q')
df1 = pd.DataFrame(df)
l = list (df1.columns)

a = []
for i in l:
    if(i[0] != ':'):
        a.append(i)
df2 = df1.loc[:,a]

df2['Gross saving'] = quandl.get('BEA/T50100_Q')['Gross saving']
b = quandl.get('BEA/T20200A_Q')[['Wages and salaries']]
c = quandl.get('BEA/T20200B_Q')[['Wages and salaries']]
d = quandl.get('BEA/T50705A_Q')[['Change in private inventories']]
e = quandl.get('BEA/T50705B_Q')[['Change in private inventories']]
df2['Wages and salaries'] = pd.concat([b,c])
df2['Change in private inventories'] = pd.concat([d,e]).drop_duplicates()
df2['National income'] = quandl.get('BEA/T11200_Q')[['National income']]
df2['Personal income'] = quandl.get('BEA/T20100_Q')[['Personal income']]
df2['Current receipts'] = quandl.get('BEA/T30300_Q')[['Current receipts']]

############################################################################
##Data Cleaning and Exploring of data
############################################################################

df2.describe()
df2.isnull()                       ###Check if there is any null values
df2.notnull()

df2 = df2.dropna()               ###drop rows with null values
df2.head()                 
df2.count()                    ###count no. of not null value in each column

df2.corr()                     ###Check corelation with each column
                               ###change in inventory is poorly co-related remove that from model

df2['Gross domestic product'].plot()                 ##timeseries plot                    
plt.show()
df2['Gross domestic product'].hist()                 ##histogram plot
plt.show()                                           
df2['Gross domestic product'].plot(kind = 'kde')     ##density plot
plt.show()                                           ##data is skewed  left side

lag_plot(df2['Gross domestic product'])              ##positive correlation relationship

autocorrelation_plot(df2['Gross domestic product'])   ##line above dotted line shows statistically significant
                                                      ##we can see trend in data and not seasonality
                                                      
##############################################################
#Multivariate regression
##############################################################

###Train- test split
y  =  df2['Gross domestic product']
x = df2.drop(['Gross domestic product'], axis = 1)

train_size = int(len(x)*0.70)
train_x,test_x = x[0:train_size],x[train_size:len(x)]
train_y,test_y = y[0:train_size],y[train_size:len(x)]

print('Observations: %d' % (len(x)))                           ##284
print('Training Observations: %d' % (len(train_x)))            ##198
print('Testing Observations: %d' % (len(test_x)))              ##86

###fit a model

lm = linear_model.LinearRegression()
model = lm.fit(train_x,train_y)

predictions_test = lm.predict(test_x)                        ##prediction on test data

lm.coef_                                                      ##check coefficient
r2_score_test = r2_score(test_y,predictions_test)             ##99.99%

####################################################
#ARIMA
####################################################

def MAPE(act,predict):
    error = abs(act - predict)
    pe = ((error/act)*100.0)
    mape = np.mean(pe)
    return(mape)
    
def AE(act,predict):
    error = abs(act - predict)
    return(error)
def MAE(act,predict):
    error = abs(act - predict)
    mae = np.mean(error)
    return(mae)

##Apply ARIMA on all factors to get forecasted values, then use this values in our regression model to get prediction of next term GDP

pred = pd.DataFrame(index=test_y.index)
MAPE_pred = []
MAE_pred = []

for i in train_x.columns:
    stepwise_fit = auto_arima(train_x[i],error_action = 'ignore',trend ='t')                            ##autoarima perform arima on different values of (p,d,q) and take which has minimum mse, because of trend in data 't'
    next_test = stepwise_fit.predict(n_periods = len(test_x))                                            ## forecast for all columns, for next preiods(length = test_length)
    pred[i] = next_test
    MAPE_pred.append(MAPE(next_test,test_x[i]))                                                           ##calculate Mean absolute percent error
    MAE_pred.append(MAE(next_test,test_x[i]))                                                       ## calculate mean absolute error
    
    
predictions = pd.DataFrame(index=test_y.index)
predictions['predicted_GDP'] = lm.predict(pred)
print mean_squared_error(test_y,predictions)                                    ##440740245580.94995
print r2_score(test_y,predictions)                                              ##0.9582            
print MAPE(test_y,predictions['predicted_GDP'])                                  ##3.63

stepwise_fit.summary()

#####################################
#Predicted vs actual for test data
######################################

plt.plot(predictions['predicted_GDP'],'-',test_y,'.')
plt.show()

##########################################################################
#Prediction for the next 2018 and 2019 GDP value
##########################################################################
test_y.tail()    # we have data till 2017-12-31
                 ## we want to predict for next 8 values, for next two year

pred2= pd.DataFrame()
date = ['2018-03-31', '2018-06-30', '2018-09-30','2018-12-31','2019-03-31', '2019-06-30', '2019-09-30','2019-12-31']
pred2['date'] = date
pred2['date'] = pd.to_datetime(pred2['date'], infer_datetime_format=True)
pred2.index = pred2['date']

pred1 = pred2.drop(['date'], axis = 1)

for i in test_x.columns:
    stepwise_fit = auto_arima(test_x[i],error_action = 'ignore',trend ='t')                            ##autoarima perform arima on different values of (p,d,q) and take which has minimum mse, because of trend in data 't'
    next_8 = stepwise_fit.predict(n_periods = 8)                                            ## forecast for all columns, for next preiods(length = test_length)
    pred1[i] = next_8
    
    
predictions_2year = pd.DataFrame(index = pred1.index)  
predictions_2year['predicted_GDP'] = lm.predict(pred1)


plt.plot(predictions_2year,'.',test_y,'-')            ##plot of forecasted values
plt.show()
