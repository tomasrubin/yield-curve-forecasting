"""
Authors : Kelly Ayliffe and Tomas Rubin, January 2020

This code compares forecasting accuracy of various yield curve models on US data.
It uses the rolling window method, with a 720 day window, to generate forecasts 1 week, 1,3,6 and 12 months in advance.
These are then compared to their real-time value. We compare each method's set of forecasts to the chosen benchmark,
the random walk, in order to determine their accuracy. This code outputs the ratios of Root Mean Square Erros and also
uses the Harvey, Leybourne and Newbold test in order to determine significant outperformances. The methods compared and their abreviations in this code are :
    (benchmark) Random walk - RW
    Two-Step Dynamic Nelson Siegel with Vector Autoregression process of lag 1 - VAR(1)+DNS
    Two-Step Dynamic Nelson Siegel with Vector Autoregression process of lag p (p >= 1 chosen by BIC criterion) - VAR(p)+DNS
    One-Step Dynamic Nelson Siegel with Kalman Filter - DNS+KF
    Traditional Autoregression process of lag 1 (AR processes seperated by maturity) - AR(1)
    Traditional 11 (= #maturities) dimensional Vector Autoregression process  of lag 1 - VAR(1)
    
The data can be found on the US treasury website : https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield

"""

#%% Package loading
import numpy as np
import math
import pandas as pd
import statsmodels.api as sm
import time
from datetime import timedelta, date
from numpy import linalg as LA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import AR
import matplotlib.pyplot as plt
from scipy.stats import t
from pykalman import KalmanFilter


#%% SECTION 0 : FUNCTION DEFINITIONS

#Computes Lambda parameter from the NS model
#INPUT: y = full data used for the estimation. tau = maturities vector
#OUTPUT: vector of lambdas computed for each cross-section. Recall the chosen lambda is the mean of this.
def lamb(y, tau):
    best_lamb=np.zeros(shape=(1,y.shape[0]))
    for i in range(0,y.shape[0]):
        current_best=1000;
        y_present = y[i]
        for j in range(1,2000):
            current_lamb = j/1000;
            fit = OLS_DNS_Error(y_present,current_lamb,tau)
            if (fit<current_best):
                current_best = fit;
                best_lamb[0,i] = current_lamb;
    return best_lamb

#Computes the error of the NS model on a cross section
#INPUT: data_i = vector of yields on one cross-section. lamb_i = lambda used in the NS model. tau_in = vector of maturities
#OUTPUT: square error of the NS model estimated by OLS with lambda = lamb_i, fitted to data_i.
def OLS_DNS_Error(data_i,lamb_i,tau_in):
    tau = tau_in.transpose()
    dummy= np.array(lamb_i*tau,dtype=float)
    col2 = (np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy
    col3 = ((np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy)-np.exp(-1*dummy)
    X = np.hstack((np.ones((tau.size,1)),col2,col3))
    est=sm.OLS(data_i, X)
    est = est.fit()
    f = est.params
    
    squ_error=0
    for i in range(tau.shape[0]):
        squ_error = squ_error + (DNS_formula(tau[i],f,lamb_i)-data_i[i])**2
    
    return squ_error
    
#Computes the NS value
#INPUT: x = vector of maturities. f = factors [l,s,c] of the NS model. lambb = lambda parameter of the NS model.
#OUTPUT: yield following the NS model from the input.
def DNS_formula(x,f,lambb):
    [l1,s1,c1]=f
    y=l1+s1*((1-np.exp(-lambb*x))/(lambb*x))+c1*((1-np.exp(-lambb*x))/(lambb*x)-np.exp(-lambb*x))
    return y

#Computes table of the OLS fitted time series [l,s,c]
#INPUT: data = yield data that needs to be fitted. tau_in = maturities vector. dat = dates of the time series. lamb_i = lambda parameter of the NS model
#OUTPUT: table with rows indexed by dat and 3 columns ([l,s,c] respectively).
def DNS_OLS(data,tau_in,lamb_i): #IMPORTANT : data is a numpy WITHOUT index on row or cols, tau_in is a horizontal vector, dat is a horizontal vector   
    tau = tau_in.transpose()
    dummy = np.array(lamb_i*tau, dtype=float)
    
    #matrix which will support the values of f=[l,s,c] at each time t
    f_concat = np.array(np.zeros(shape = (data.shape[0],3)));
    
    #Static Nelson Siegel for each time t
    for i in range(0,y.shape[0]):
        #selection of the current data
        y_i = np.array([data[i]]).transpose()

        #Computation of the least squares estimator for f    
        col2 = (np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy
        col3 = ((np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy)-np.exp(-1*dummy)
        
        X = np.hstack((np.ones((tau.size,1)),col2,col3))

        #OLS solution :      
        est=sm.OLS(y_i, X)
        est = est.fit()
        
        f = est.params
        f_concat[i] = f

    return f_concat

#Chosen loss function
#INPUT : n=0 : squared error / n=1 : absolute error
#OUTPUT : g(e_it)
def g(e_it,n=0):
    v=0
    if (n==0):
        v=e_it**2
    if (n==1):
        v=abs(e_it)
    return v

#Forecasting with last known value
#INPUT : da = sample of data used for the estimation. pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the yield curve forecast i days ahead of the last date in da
def forecast_RW_fct(da,pred=1):
    forecast = np.zeros(shape=(pred,da.shape[1]))
    for i in range(0,da.shape[1]):
        forecast[:,i] = np.ones(shape=(pred))*da[da.shape[0]-1,i]
    return forecast

def params_VAR(da):
    model = VAR(da)
    lag_order=1
    model_fitted = model.fit(lag_order)
    trans_matrix = (model_fitted.params).to_numpy()
    return trans_matrix

#Forecasting with two-step DNS, VAR(1).
#INPUT : ts = table of OLS fitted factors (l,s,c) per cross-section (they are generated by ts_DNS_creator_daily.py). pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the factor forecast i days ahead of the last date in ts.
def forecast_DNS_VAR(ts, pred): #IMPORTANT : ts has undergone the DNS_OLS function previously. pred is the date pred months after the last entry of the time series ts

    model = VAR(ts)
    model_fitted = model.fit(1) #See Diebold and Rudebusch. All use VAR(1)
    
    lag_order=model_fitted.k_ar
    
    return model_fitted.forecast(ts.values[-lag_order:],pred)

#Forecasting with two-step DNS, VAR(p).
#INPUT : ts = table of OLS fitted factors (l,s,c) per cross-section (they are generated by ts_DNS_creator_daily.py). pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the factor forecast i days ahead of the last date in ts.
def forecast_DNS_VARm(ts,pred):
    model = VAR(ts)
    x = model.select_order(maxlags=3)
    lag_order = x.selected_orders["bic"] #we select best model based on the BIC criterion
    if lag_order==0:    #constrains not turning into a random walk
        lag_order=1
    model_fitted = model.fit(lag_order)
    return model_fitted.forecast(ts.values[-lag_order:],pred)

#Forecasting with VAR(1).
#INPUT : da = sample of data used for the estimation. pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the yield curve forecast i days ahead of the last date in da.
def forecast_VAR(da, pred):
    model = VAR(da)
    lag_order = 1
    model_fitted = model.fit(lag_order)
    return model_fitted.forecast(da.values[-lag_order:],pred)

#Forecasting with AR(1).
#INPUT : da = sample of data used for the estimation. pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the yield curve forecast i days ahead of the last date in da.
def forecast_AR(da, pred):
    forecast = np.zeros(shape=(pred,da.shape[1]))
    #forecasting per maturity (suppose no dependance between maturities)
    for i in range(0,da.shape[1]):
        #fit to time series y_{tau} for fixed tau
        model = AR(da[:,i])
        model_fitted = model.fit(1)
        
        #assume y_{tau,t+1} = b0 + b1*y_{tau,t} + \epsilon
        b0=model_fitted.params[0]
        b1=model_fitted.params[1]
        
        #compute predictions by rolling out the recursion : \hat{y}_{tau,t+j} = b0*(1+...+b1^{j-1}) + b1^j * \hat{y}_{tau,t} 
        for j in range(0,pred):
            for k in range(j,pred-1):
                forecast[k,i-1]=forecast[k,i-1]+b0*b1**k
            forecast[j,i-1]=forecast[j,i-1] + b1**j*da[da.shape[0]-1,i]
    return forecast

#Forecasting with one-step DNS
#INPUT : da = sample of data used for the estimation. tau = vector of maturities. lamb = lambda factor estimated. state_init = initialisation of the states means for KF .pred =  number of forecasts ahead (in days)
#OUTPUT : table with pred rows. Row i is the yield curve forecast i days ahead of the last date in ts.
def forecast_DNS_KF(da,tau,lamb,state_init,offset_init,trans_init,pred):
    t = np.array(tau)
    t = t.transpose()
    
    dat = da.to_numpy()
    
    #STATE EQUATION
    #-- state initialisation
    initial_state_mean = state_init
    m = np.random.normal(0,1,[3,3])
    initial_state_covariance = m@m.transpose() # to achieve positive definite
    
    #-- transitions initialisation
    initial_transition_matrix = trans_init
    initial_transition_offset = offset_init
    m = np.random.normal(0,1,[3,3])
    initial_transition_covariance = m@m.transpose()
    

    #OBSERVATION EQUATION
    #Transition Matrix X is determined by Nelson Siegel dynamics
    dummy = np.array(lamb*t, dtype=float)
    
    col2 = (np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy
    col3 = ((np.ones(shape=(tau.size,1))-np.exp(-1*dummy))/dummy)-np.exp(-1*dummy)

    
    X = np.hstack((np.ones((tau.size,1)),col2,col3))
    
    true_observation_matrices = X
    true_observation_offsets = np.zeros(shape=(dat.shape[1]))
    
    m = np.random.normal(0,1,[dat.shape[1],dat.shape[1]])
    initial_observation_covariance = m@m.transpose()
    
    #-- Uncomment for a non-random initialisation of covariance matrices
    #random_init_observation_covariance = 0.1*np.eye(dat.shape[1])
    #initial_state_covariance = 0.1*np.eye(3)
    #initial_transition_covariance = 0.1*np.eye(3)
    
    
    #Kalman Filter estimation
    kf = KalmanFilter(transition_matrices=initial_transition_matrix,\
                      observation_matrices=true_observation_matrices,\
                      transition_covariance=initial_transition_covariance,\
                      observation_covariance=initial_observation_covariance,\
                      transition_offsets=initial_transition_offset,\
                      observation_offsets=true_observation_offsets,\
                      initial_state_mean = initial_state_mean,\
                      initial_state_covariance=initial_state_covariance,\
                      em_vars=['transition_matrices', 'transition_offsets','transition_covariance', 'observation_covariance',\
                               'initial_state_mean','initial_state_covariance'],\
                      n_dim_state=3,\
                      n_dim_obs=dat.shape[1])
    kf = kf.em(dat, n_iter=20) #EM estimation of quantities specified in em_vars
    smoother_x, smoother_var = kf.smooth(dat) #KF smoother estimation of the states
    
    #Uncomment for plots of [l,s,c] smoother estimations
    """
    fig = plt.figure()
    plt.plot( smoother_x[:,0] )
    plt.legend(['smoother x, l state'])
    plt.show()
    
    fig = plt.figure()
    plt.plot( smoother_x[:,1] )
    plt.legend(['smoother x, s state'])
    plt.show()
    
    fig = plt.figure()
    plt.plot( smoother_x[:,2] )
    plt.legend(['smoother x, c state'])
    plt.show()
    """
    # recall we had f_{t+1} = transition_matrices*f_{t} + transition_offsets + epsilon where epsilon ~ N (0, transition_covariance)
    # therefore forecasting h days ahead is done as is f*_{t+1} = (transition_matrices)*f_{t} + transition_offsets
    forecast=np.zeros(shape=(pred,3))
    last_value = np.array([smoother_x[-1]])
    offs = kf.transition_offsets
    for i in range(pred):
        forecast[i,:] = np.array(last_value @ kf.transition_matrices.transpose() + offs)
        last_value = forecast[i,:]
    return forecast

#Sample autocovariance at lag k
#INPUT:  k = lag, d = time series
#OUTPUT: sample autocovarance of time series d at lag k
def gamh(k,d):
    gam = 0
    db = np.mean(d)
    for tt in range((abs(k)+1),(d.shape[0])):
        gam = gam + (d[tt]-db)*(d[tt-abs(k)]-db)    
    
    gam = gam/(d.shape[0])
    return gam

#HLN test statistic
#INPUT: y1,y2 = vectors of forecasts of the two methods to compare (here y1 -> RW). y = vector of corresponding true value. h = number of steps ahead of the forecast
#OUTPUT: HLN test statistic
def DM(y1,y2,y,h):
    dm = 0
    if (y1.shape==y2.shape) and (y1.shape==y.shape):
        e_1 = y1 - y
        e_2 = y2 - y
        T = y.shape[0]
        d = g(e_1) - g(e_2)
        dbar = np.mean(d)
        fh0 = gamh(0,d)
        M = int(math.floor(math.pow(T,1/3))+1)
        for k in range(-M,M):
            fh0 = fh0 + 2*gamh(k,d)
        fh0 = fh0*(1/(2*math.pi))
        dm = dbar/(math.pow((2*math.pi*fh0)/T,1/2))
        hln=math.pow((T+1-2*h+h*(h-1))/T,1/2)*dm
        return hln
    else: 
        return -10000

#Student Test
#INPUT: tval = table with sample values of the test statistic. n = degrees of freedom.
#OUTPUT: two tables.
#Table 1: p value table ("-" in front of the p_value indicates that p-value is for RW significantly better, "+" sign for method better)
#Table 2: indicator table. entry = sign(test_stat). +1: method is (maybe) better, -1: RW is (maybe) better -> see Table 1 for p-values
def student_test(tval,n):
    
    ptab=np.zeros(shape=(tval.shape[0],tval.shape[1]))
    qtab=np.zeros(shape=(tval.shape[0],tval.shape[1]))
    
    for i in range(ptab.shape[0]):
        for j in range(ptab.shape[1]):

            if (tval[i][j]>0):  #a positive value means the method performs better than RW
                ptab[i][j] = t.cdf(-tval[i][j],n,0,1)
                if(t.cdf(-tval[i][j],n,0,1)<=0.05): qtab[i][j] = 1
                
            if (tval[i][j]<0): #a negative value means the method performs worse than the RW
                ptab[i][j] = -t.cdf(tval[i][j],n,0,1)
                if(t.cdf(tval[i][j],n,0,1)<=0.05): qtab[i,j] = -1
    return np.hstack((ptab,qtab))


#%% SECTION 1 : fills the missing values in the data (data only available on business days) by the last known available value.
"""
Note on the Excel file : the maturities of the yields should be specified in the first row as the column index, 
The dates of the yields should be specified in the first column as the row index. The ones here are the US Treasury yield maturities.
"""
y_df = pd.read_excel('US_daily.xlsx', columns = ['dates',1/12,3/12,6/12,1,2,3,5,7,10,20,30], index='dates');
y = y_df.to_numpy()
matu = np.array([[1/12,3/12,6/12,1,2,3,5,7,10,20,30]])

dates = y[:,0]

current=1 #this variable keeps track of the dates in the original dataset that have already been added. It is a row index in the original table.
currentDate = np.datetime64(dates[0]) # this variable keeps track of all dates that need to be added.

# The following two tables will be concatenated horizontally to create the full, new dataset
CompleteTable = np.array([y[0,1:]]) #Table with added yields (has copied lines where extra dates have been added)
CompleteDates = np.array([[currentDate]], dtype='datetime64') #Will be the full dates column

AddDay = np.timedelta64(1,'D')

cdnp = np.array([[currentDate]],dtype='datetime64') #single entry array. Used to have a compatible format (np.array) for adding the dates to CompleteDates.

while current<y_df.shape[0]:
    currentDate = currentDate + AddDay
    cdnp[0][0] = currentDate
    CompleteDates = np.hstack((CompleteDates,cdnp))
    dateInTable = np.datetime64(dates[current])
    
    if dateInTable != currentDate:
        CompleteTable = np.vstack((CompleteTable,CompleteTable[-1])) #copies last available line into the table
        
    if dateInTable == currentDate:
        CompleteTable = np.vstack((CompleteTable,y[current,1:])) #adds yield curve corresponding to currentDate
        current = current + 1

#Updating to full table
y = np.hstack((CompleteDates.transpose(), CompleteTable))
dates = np.array([y[:,0]])
y = np.delete(y,0,1) #seperating dates and yields
y = np.array(y,dtype = float)

#%% SECTION 2 : computes the coefficients of the Static Nelson Siegel curve for each date (fits the NS parametrisation with a fixed lamda using OLS)

# Lambda computation
"""
Note on the lambda computation : we compute the optimal (in the least squares sense) lambda per cross section by grid search and
take the mean over all cross sections in order to have a lambda for the full model.
"""
#Uncomment below for full lambda computation
"""
our_lambda_vector = lamb(y,matu) #see function definition in section 0 above
our_lambda = float(np.mean(our_lambda_vector))

#Descriptive statistics for the fitted lambdas
stats_lambda = np.zeros(shape=(1,4))
stats_lambda[0,0] = our_lambda
stats_lambda[0,1] = np.std(our_lambda_vector)
stats_lambda[0,2] = np.amin(our_lambda_vector)
stats_lambda[0,3] = np.amax(our_lambda_vector)

print('Mean / St Dev / Min / Max ', stats_lambda, '\n')
print('Selected lambda : ', our_lambda)
"""
our_lambda = 0.496 #result of commented computation

# OLS fitting of the coefficients
ts = DNS_OLS(y,matu,our_lambda)
tsf = pd.DataFrame(ts) #some functions require the data in Pandas instead of Numpy

#Uncomment to obtain plots of OLS fits 
"""
for i in range(0,ts.shape[0]):
# Prepare the data
    x_plot1 = np.linspace(0, 30, 400)
    l1=ts[i,0]
    s1=ts[i,1]
    c1=ts[i,2]
    y_plot1 = l1+s1*((1-np.exp(-our_lambda*x_plot1))/(our_lambda*x_plot1))+c1*((1-np.exp(-our_lambda*x_plot1))/(our_lambda*x_plot1)-np.exp(-our_lambda*x_plot1))
    
    # Plot the data
    plt.plot(x_plot1, y_plot1, label='DNS plot')
    plt.scatter(matu,y[i,:])
    # Add a legend
    plt.legend(('plot of NS curve with fitted OLS values','actual values'))
    # Show the plot
    plt.show()
"""

#%% SECTION 3 : THE TRAVELLING WINDOW

p = matu.shape[1] #p = nb maturities
nb_dates = y.shape[0]

#The following tables stock forecasts. Each row of forecasts_METHOD_XX is the forecast produced by METHOD, XX in advance (w = week, m = month)
#We subtract 1080 = 720 + 360 as our window is 720 days long and the maximal forecasting horizon is 360 days.

#Two step DNS with VAR(1) process modelling the extracted factors
forecasts_VAR_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VAR_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VAR_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_VAR_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_VAR_m12= np.zeros(shape=(nb_dates-1080,p))

# Two step DNS with VAR(p) process modelling the extracted factors
forecasts_VARm_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARm_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARm_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARm_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARm_m12= np.zeros(shape=(nb_dates-1080, p))

#VAR(p) process without DNS (models the data as a p-dimensional time series)
forecasts_VARn_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARn_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARn_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARn_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_VARn_m12= np.zeros(shape=(nb_dates-1080, p))

#AR(p) process without DNS (models the data as p seperate time series)
forecasts_ARn_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_ARn_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_ARn_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_ARn_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_ARn_m12= np.zeros(shape=(nb_dates-1080, p))

#One step DNS, with the Kalman Filter + VAR(1)
forecasts_KF_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_KF_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_KF_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_KF_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_KF_m12= np.zeros(shape=(nb_dates-1080, p))

#Random walk process
forecasts_RW_w1= np.zeros(shape=(nb_dates-1080, p))
forecasts_RW_m1= np.zeros(shape=(nb_dates-1080, p))
forecasts_RW_m3= np.zeros(shape=(nb_dates-1080, p))
forecasts_RW_m6= np.zeros(shape=(nb_dates-1080, p))
forecasts_RW_m12= np.zeros(shape=(nb_dates-1080, p))

#Actual values : real time values of the yields we have forecasted above. This will be used to determine the accuracy of the forecast.
actual_w1 = np.zeros(shape=(nb_dates-1080, p))
actual_m1 = np.zeros(shape=(nb_dates-1080, p))
actual_m3 = np.zeros(shape=(nb_dates-1080, p))
actual_m6 = np.zeros(shape=(nb_dates-1080, p))
actual_m12 = np.zeros(shape=(nb_dates-1080, p))

for i in range(nb_dates-1080):
    data = y[i:i+720,:]  #selects window of 2 years where 1 year is 360 days (finanical convention)
    dataf = pd.DataFrame(data)
    actual_w1[i] = y[i+727,:]
    actual_m1[i] = y[i+750,:] #consider 1 month has 30 days
    actual_m3[i] = y[i+810,:]
    actual_m6[i] = y[i+900,:]
    actual_m12[i] = y[i+1080,:]
    
    step_one = ts[i:i+720] #selects the window of the time series [L,S,C] generated by OLS fitting
    step_onef = tsf[i:i+720] #some functions require the data in Pandas instead of Numpy
    forecast_RW = forecast_RW_fct(data,360) #Random walk
    
    step_two_VAR = forecast_DNS_VAR(step_onef,360) #Two Step DNS with VAR(1)
    
    step_two_VARm = forecast_DNS_VARm(step_onef, 360) #Two Step DNS with VAR(p)
    
    forecast_VARn = forecast_VAR(dataf,360)
    
    forecast_ARn = forecast_AR(data,360)

    #We choose to initialise the EM algorithm with VAR fitted parameters, because they are likely to make it converge to the global maximum.
    state_init = step_one[0]
    params_init = params_VAR(tsf)
    offset_init = params_init[0,:]
    transition_init = np.array(params_init[1:,:]).transpose()

    forecast_KF = forecast_DNS_KF(dataf,matu,our_lambda,state_init,offset_init,transition_init,360) #One Step DNS with VAR(1).
    
    #Selecting the 1 week, 1 month, 3 months, 6 months, 12 months ahead forecasts
    forecasts_RW_w1[i]= forecast_RW[6]
    forecasts_RW_m1[i]= forecast_RW[29]
    forecasts_RW_m3[i]= forecast_RW[89]
    forecasts_RW_m6[i]= forecast_RW[179]
    forecasts_RW_m12[i]= forecast_RW[359]
    
    #DNS+VAR(1)
    [l_iw1,s_iw1,c_iw1] = step_two_VAR[6]
    [l_i1,s_i1,c_i1] = step_two_VAR[29]
    [l_i3,s_i3,c_i3] = step_two_VAR[89]
    [l_i6,s_i6,c_i6] = step_two_VAR[179]
    [l_i12,s_i12,c_i12] = step_two_VAR[359]
    
    forecasts_VAR_w1[i] = l_iw1*np.ones(matu.shape) + s_iw1*((np.ones(matu.shape)-np.exp((-1)*our_lambda*matu))/(our_lambda*matu))  + c_iw1*(((np.ones(matu.shape)-np.exp((-1)*our_lambda*matu))/(our_lambda*matu))-np.exp((-1)*our_lambda*matu))
    forecasts_VAR_m1[i] = l_i1*np.ones(matu.shape) + s_i1*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i1*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VAR_m3[i] = l_i3*np.ones(matu.shape) + s_i3*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i3*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VAR_m6[i] = l_i6*np.ones(matu.shape) + s_i6*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i6*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VAR_m12[i] = l_i12*np.ones(matu.shape) + s_i12*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i12*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))

    #DNS + VAR(p)
    [l_iw1,s_iw1,c_iw1] = step_two_VARm[6]
    [l_i1,s_i1,c_i1] = step_two_VARm[29]
    [l_i3,s_i3,c_i3] = step_two_VARm[89]
    [l_i6,s_i6,c_i6] = step_two_VARm[179]
    [l_i12,s_i12,c_i12] = step_two_VARm[359]
        
    forecasts_VARm_w1[i] = l_iw1*np.ones(matu.shape) + s_iw1*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_iw1*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VARm_m1[i] = l_i1*np.ones(matu.shape) + s_i1*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i1*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VARm_m3[i] = l_i3*np.ones(matu.shape) + s_i3*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i3*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VARm_m6[i] = l_i6*np.ones(matu.shape) + s_i6*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i6*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_VARm_m12[i] = l_i12*np.ones(matu.shape) + s_i12*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i12*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
   
    #VAR(1) 
    forecasts_VARn_w1[i]= forecast_VARn[6]
    forecasts_VARn_m1[i]= forecast_VARn[29]
    forecasts_VARn_m3[i]= forecast_VARn[89]
    forecasts_VARn_m6[i]= forecast_VARn[179]
    forecasts_VARn_m12[i]= forecast_VARn[359]
    
    #AR(1)
    forecasts_ARn_w1[i]= forecast_ARn[6]
    forecasts_ARn_m1[i]= forecast_ARn[29]
    forecasts_ARn_m3[i]= forecast_ARn[89]
    forecasts_ARn_m6[i]= forecast_ARn[179]
    forecasts_ARn_m12[i]= forecast_ARn[359]
        
    #DNS+KF
    [l_iw1,s_iw1,c_iw1] = forecast_KF[6]
    [l_i1,s_i1,c_i1] = forecast_KF[29]
    [l_i3,s_i3,c_i3] = forecast_KF[89]
    [l_i6,s_i6,c_i6] = forecast_KF[179]
    [l_i12,s_i12,c_i12] = forecast_KF[359]
    
    
    forecasts_KF_w1[i] = l_iw1*np.ones(matu.shape) + s_iw1*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_iw1*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_KF_m1[i] = l_i1*np.ones(matu.shape) + s_i1*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i1*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_KF_m3[i] = l_i3*np.ones(matu.shape) + s_i3*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i3*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_KF_m6[i] = l_i6*np.ones(matu.shape) + s_i6*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i6*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))
    forecasts_KF_m12[i] = l_i12*np.ones(matu.shape) + s_i12*((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))  + c_i12*(((np.ones(matu.shape)-np.exp(-our_lambda*matu))/(our_lambda*matu))-np.exp(-our_lambda*matu))

    #Uncomment to plot the comparison of 6 month ahead forecasts with One Step and Two Step DNS methods to RW. 
    """
    print('This is try number')
    print(i)
    plot_matu = matu.transpose()
    fig = plt.figure()                
    plt.plot(plot_matu, actual_m6[i],color="green")        
    plt.plot(plot_matu, data[-1,:], color="blue")
    plt.scatter(plot_matu,forecasts_VAR_m6[i],color="red")
    plt.scatter(plot_matu,forecasts_KF_m6[i],color="purple")
    plt.scatter(plot_matu,forecasts_RW_m6[i], color="red", marker="x" )
    plt.legend(('true yield curve (tb forecasted)','last y curve (for RW)','forecast 2 step DNS','forecast KF',"forecast RW"))
    plt.show()
    
    fig = plt.figure()                
    plt.plot(plot_matu, actual_m12[i],color="green")        
    plt.plot(plot_matu, data[-1,:], color="blue")
    plt.scatter(plot_matu,forecasts_VAR_m12[i],color="red")
    plt.scatter(plot_matu,forecasts_KF_m12[i],color="purple")
    plt.scatter(plot_matu,forecasts_RW_m12[i], color="red", marker="x" )
    plt.legend(('true yield curve (tb forecasted)','last y curve (for RW)','forecast 2 step DNS','forecast KF',"forecast RW"))
    plt.show()
    """

#%% SECTION 4 : COMPARING THE METHODS WITH THE HLN TEST AND RMSE RATIOS
"""
RATIO TABLES
Each method is compared with the RW as follows : Ratio > 1 => RMSE(considered method) > RMSE(RW) => RW better than considered method

In order to statistically determine whether the outperformance is significant, we use the Harvey, Leybourne and Newbold (HLN) test.

"""

#Computing Root Mean Squared Error Ratios

#Tables of actual RMSE
MSE_w1 = np.zeros(shape=(p,6))
MSE_m1 = np.zeros(shape=(p,6))
MSE_m3 = np.zeros(shape=(p,6))
MSE_m6 = np.zeros(shape=(p,6))
MSE_m12 = np.zeros(shape=(p,6))

#Tables of RMSE ratios between method and RW
ratio_w1 = np.zeros(shape=(p,5))
ratio_m1 = np.zeros(shape=(p,5))
ratio_m3 = np.zeros(shape=(p,5))
ratio_m6 = np.zeros(shape=(p,5))
ratio_m12 = np.zeros(shape=(p,5))

for i in range(0,p): #i = maturity index

    #RW
    MSE_w1[i,5] = np.mean((forecasts_RW_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,5] = np.mean((forecasts_RW_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,5] = np.mean((forecasts_RW_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,5] = np.mean((forecasts_RW_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,5] = np.mean((forecasts_RW_m12[:,i] - actual_m12[:,i])**2)**0.5
       
    #DNS+VAR(1)
    MSE_w1[i,0] = np.mean((forecasts_VAR_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,0] = np.mean((forecasts_VAR_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,0] = np.mean((forecasts_VAR_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,0] = np.mean((forecasts_VAR_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,0] = np.mean((forecasts_VAR_m12[:,i] - actual_m12[:,i])**2)**0.5

    #ratio > 1 => RMSE(considered method) > RMSE(RW) => RW better than considered method
    ratio_w1[i,0] = MSE_w1[i,0] / MSE_w1[i,5]
    ratio_m1[i,0] = MSE_m1[i,0] / MSE_m1[i,5]
    ratio_m3[i,0] = MSE_m3[i,0] / MSE_m3[i,5]
    ratio_m6[i,0] = MSE_m6[i,0] / MSE_m6[i,5]
    ratio_m12[i,0] = MSE_m12[i,0] / MSE_m12[i,5]
    
    #DNS+VAR(p)
    MSE_w1[i,1] = np.mean((forecasts_VARm_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,1] = np.mean((forecasts_VARm_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,1] = np.mean((forecasts_VARm_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,1] = np.mean((forecasts_VARm_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,1] = np.mean((forecasts_VARm_m12[:,i] - actual_m12[:,i])**2)**0.5
    
    ratio_w1[i,1] = MSE_w1[i,1] / MSE_w1[i,5]
    ratio_m1[i,1] = MSE_m1[i,1] / MSE_m1[i,5]
    ratio_m3[i,1] = MSE_m3[i,1] / MSE_m3[i,5]
    ratio_m6[i,1] = MSE_m6[i,1] / MSE_m6[i,5]
    ratio_m12[i,1] = MSE_m12[i,1] / MSE_m12[i,5]
    
    #DNS+KF
    MSE_w1[i,2] = np.mean((forecasts_KF_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,2] = np.mean((forecasts_KF_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,2] = np.mean((forecasts_KF_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,2] = np.mean((forecasts_KF_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,2] = np.mean((forecasts_KF_m12[:,i] - actual_m12[:,i])**2)**0.5

    ratio_w1[i,2] = MSE_w1[i,2] / MSE_w1[i,5]
    ratio_m1[i,2] = MSE_m1[i,2] / MSE_m1[i,5]
    ratio_m3[i,2] = MSE_m3[i,2] / MSE_m3[i,5]
    ratio_m6[i,2] = MSE_m6[i,2] / MSE_m6[i,5]
    ratio_m12[i,2] = MSE_m12[i,2] / MSE_m12[i,5]
        
    #AR(1)
    MSE_w1[i,3] = np.mean((forecasts_ARn_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,3] = np.mean((forecasts_ARn_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,3] = np.mean((forecasts_ARn_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,3] = np.mean((forecasts_ARn_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,3] = np.mean((forecasts_ARn_m12[:,i] - actual_m12[:,i])**2)**0.5
    
    ratio_w1[i,3] = MSE_w1[i,3] / MSE_w1[i,5]
    ratio_m1[i,3] = MSE_m1[i,3] / MSE_m1[i,5]
    ratio_m3[i,3] = MSE_m3[i,3] / MSE_m3[i,5]
    ratio_m6[i,3] = MSE_m6[i,3] / MSE_m6[i,5]
    ratio_m12[i,3] = MSE_m12[i,3] / MSE_m12[i,5]
    
    #VAR(1)
    MSE_w1[i,4] = np.mean((forecasts_VARn_w1[:,i] - actual_w1[:,i])**2)**0.5
    MSE_m1[i,4] = np.mean((forecasts_VARn_m1[:,i] - actual_m1[:,i])**2)**0.5
    MSE_m3[i,4] = np.mean((forecasts_VARn_m3[:,i] - actual_m3[:,i])**2)**0.5
    MSE_m6[i,4] = np.mean((forecasts_VARn_m6[:,i] - actual_m6[:,i])**2)**0.5
    MSE_m12[i,4] = np.mean((forecasts_VARn_m12[:,i] - actual_m12[:,i])**2)**0.5
    
    ratio_w1[i,4] = MSE_w1[i,4] / MSE_w1[i,5]
    ratio_m1[i,4] = MSE_m1[i,4] / MSE_m1[i,5]
    ratio_m3[i,4] = MSE_m3[i,4] / MSE_m3[i,5]
    ratio_m6[i,4] = MSE_m6[i,4] / MSE_m6[i,5]
    ratio_m12[i,4] = MSE_m12[i,4] / MSE_m12[i,5]
    

print('\n\n\nACTUAL MSEs : \n')
print('1 week forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1)/RW')
print(MSE_w1)
print('\n')
    
print('1 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1)/RW')
print(MSE_m1)
print('\n')

print('3 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1)/RW \n')
print(MSE_m3)
print('\n')

print('6 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1)/RW \n')
print(MSE_m6)
print('\n')

print('12 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1)/RW \n')
print(MSE_m12)
print('\n')

print('\n\n')

print('RATIOS WITH RANDOM WALK : \n')

print('1 week forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(ratio_w1)
print('\n')

print('1 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(ratio_m1)
print('\n')

print('3 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(ratio_m3)
print('\n')

print('6 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(ratio_m6)
print('\n')

print('12 month forward for each maturity : OLS+VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(ratio_m12)
print('\n')


# (Diebold Mariano) HLN Test

nb_methods = 5

#Tables with test-statistic values. 
HLN_w1 = np.zeros(shape=(p,nb_methods))
HLN_m1 = np.zeros(shape=(p,nb_methods))
HLN_m3 = np.zeros(shape=(p,nb_methods))
HLN_m6 = np.zeros(shape=(p,nb_methods))
HLN_m12 = np.zeros(shape=(p,nb_methods))


for i in range(p):
    
    #DNS+VAR(1)
    HLN_w1[i,0] = DM(forecasts_RW_w1[:,i], forecasts_VAR_w1[:,i], actual_w1[:,i],7)
    HLN_m1[i,0] = DM(forecasts_RW_m1[:,i], forecasts_VAR_m1[:,i], actual_m1[:,i],30)
    HLN_m3[i,0] = DM(forecasts_RW_m3[:,i], forecasts_VAR_m3[:,i], actual_m3[:,i],90)
    HLN_m6[i,0] = DM(forecasts_RW_m6[:,i], forecasts_VAR_m6[:,i], actual_m6[:,i],180)
    HLN_m12[i,0] = DM(forecasts_RW_m12[:,i], forecasts_VAR_m12[:,i], actual_m12[:,i],360)

    #DNS + VAR(p)
    HLN_w1[i,1] = DM(forecasts_RW_w1[:,i], forecasts_VARm_w1[:,i], actual_w1[:,i],7)
    HLN_m1[i,1] = DM(forecasts_RW_m1[:,i], forecasts_VARm_m1[:,i], actual_m1[:,i],30)
    HLN_m3[i,1] = DM(forecasts_RW_m3[:,i], forecasts_VARm_m3[:,i], actual_m3[:,i],90)
    HLN_m6[i,1] = DM(forecasts_RW_m6[:,i], forecasts_VARm_m6[:,i], actual_m6[:,i],180)
    HLN_m12[i,1] = DM(forecasts_RW_m12[:,i], forecasts_VARm_m12[:,i], actual_m12[:,i],360)
    
    #DNS + KF
    HLN_w1[i,2] = DM(forecasts_RW_w1[:,i], forecasts_KF_w1[:,i], actual_w1[:,i],7)
    HLN_m1[i,2] = DM(forecasts_RW_m1[:,i], forecasts_KF_m1[:,i], actual_m1[:,i],30)
    HLN_m3[i,2] = DM(forecasts_RW_m3[:,i], forecasts_KF_m3[:,i], actual_m3[:,i],90)
    HLN_m6[i,2] = DM(forecasts_RW_m6[:,i], forecasts_KF_m6[:,i], actual_m6[:,i],180)
    HLN_m12[i,2] = DM(forecasts_RW_m12[:,i], forecasts_KF_m12[:,i], actual_m12[:,i],360)

    #AR(1)
    HLN_w1[i,3] = DM(forecasts_RW_w1[:,i], forecasts_ARn_w1[:,i], actual_w1[:,i],7)
    HLN_m1[i,3] = DM(forecasts_RW_m1[:,i], forecasts_ARn_m1[:,i], actual_m1[:,i],30)
    HLN_m3[i,3] = DM(forecasts_RW_m3[:,i], forecasts_ARn_m3[:,i], actual_m3[:,i],90)
    HLN_m6[i,3] = DM(forecasts_RW_m6[:,i], forecasts_ARn_m6[:,i], actual_m6[:,i],180)
    HLN_m12[i,3] = DM(forecasts_RW_m12[:,i], forecasts_ARn_m12[:,i], actual_m12[:,i],360)

    #VAR(1)
    HLN_w1[i,4] = DM(forecasts_RW_w1[:,i], forecasts_VARn_w1[:,i], actual_w1[:,i],7)
    HLN_m1[i,4] = DM(forecasts_RW_m1[:,i], forecasts_VARn_m1[:,i], actual_m1[:,i],30)
    HLN_m3[i,4] = DM(forecasts_RW_m3[:,i], forecasts_VARn_m3[:,i], actual_m3[:,i],90)
    HLN_m6[i,4] = DM(forecasts_RW_m6[:,i], forecasts_VARn_m6[:,i], actual_m6[:,i],180)
    HLN_m12[i,4] = DM(forecasts_RW_m12[:,i], forecasts_VARn_m12[:,i], actual_m12[:,i],360)    
    
#Corresponding P-Value
T = actual_m1.shape[0] #degrees of freedom (see HLN test)

pv_w1 = student_test(HLN_w1,T-1)
pv_m1 = student_test(HLN_m1,T-1)
pv_m3 = student_test(HLN_m3,T-1)
pv_m6 = student_test(HLN_m6,T-1)
pv_m12 = student_test(HLN_m12,T-1)
    
print('TEST STATISTIC FOR COMPARISON WITH RANDOM WALK : \n reminder : under the hypothesis that the two performances are equivalent, this value asymptotically follows a normal distribution.\n')

print('1 week forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(HLN_w1)
print('\n')

print('1 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(HLN_m1)
print('\n')

print('3 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(HLN_m3)
print('\n')

print('6 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(HLN_m6)
print('\n')

print('12 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(HLN_m12)
print('\n')

print('P-VALUES AND INDICATORS OF HLN TEST : \n')
print('indicator table legend (5% significance level): \n -1 => the method does not outperform RW \n 0 => it cannot be determined if the method outperforms RW \n 1 => the method outperforms the RW \n')

print('1 week forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(pv_w1[:,0:nb_methods])
print(pv_w1[:,nb_methods:])
print('\n')

print('1 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(pv_m1[:,0:nb_methods])
print(pv_m1[:,nb_methods:])
print('\n')

print('3 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(pv_m3[:,0:nb_methods])
print(pv_m3[:,nb_methods:])
print('\n')

print('6 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(pv_m6[:,0:nb_methods])
print(pv_m6[:,nb_methods:])
print('\n')

print('12 month forward for each maturity : VAR(1)+DNS/VAR(p)+DNS/KF+DNS/AR(1)/VAR(1) \n')
print(pv_m12[:,0:nb_methods])
print(pv_m12[:,nb_methods:])
print('\n')