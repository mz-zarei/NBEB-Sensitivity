#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 12 Dec 2019
# ---------------------------------------------------------------------------
"""Implentation of empirical Bayes estimation using NB model"""
# ---------------------------------------------------------------------------
# Imports

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def simulateCrashData(coef_vector, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a NB distributed data set n Data Frame with given coefficients, size in a linear functional form. 
    The features are uniformly distributed. 
    '''

    Obs, features, Lambda, feature_name = [], [], [], []
    coef_vector = np.array(coef_vector)
    
    for i in range(data_size):
          
        X = np.random.uniform(0,1,size=len(coef_vector))


        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)     

        ro = np.exp(np.dot(coef_vector,X) + constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))
        features.append(X)
    
    for i in range(1,len(coef_vector)+1):
        feature_name.append('X'+str(i))
    simulated_data = pd.DataFrame(features, columns = feature_name)
    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data

def simulateCrashData_NL(coef_vector, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a NB distributed data set n Data Frame with given coefficients, size in a non-linear functional form. 
    The features are uniformly distributed. 
    '''

    Obs, features, Lambda, feature_name = [], [], [], []
    coef_vector = np.array(coef_vector)
    
    for i in range(data_size):
          
        X = np.random.uniform(0,1,size=len(coef_vector))


        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)  
        X_nl = [X[0]**2, X[0]*X[1], X[2]**0.5, (X[3]*X[1])**0.25 ]

        ro = np.exp(np.dot(coef_vector,X_nl) + constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))
        features.append(X)
    
    for i in range(1,len(coef_vector)+1):
        feature_name.append('X'+str(i))
    simulated_data = pd.DataFrame(features, columns = feature_name)
    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data


def simulateCrashData_X(X_data, coef_vector, replacement = False, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a NB distributed data set n Data Frame with given an X dataframe, coefficients, size in a linear functional form. 
    The features are uniformly distributed. 
    '''
    simulated_data = pd.DataFrame(X_data.sample(n=data_size, replace=replacement))
    simulated_data.reset_index(inplace = True, drop = True)
    Obs, Lambda = [], []
    coef_vector = np.array(coef_vector)  


    for X in simulated_data.values:

        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)     

        ro = np.exp(np.dot(coef_vector,X)+ constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))

    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data

def computeDisperssion(data, features, y_name='Obs'):
    '''
    Returns disperssion parameter and corresponding SE with for given crash dataframe
    '''

    result = data.copy()
    result.reset_index(inplace=True, drop=True)
    X = sm.add_constant(data[features])
    y =  data[y_name]

    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    result['lambda'] = poisson_model.mu
    result['AUX_OLS_DEP'] = result.apply(lambda x: ((x[y_name] - x['lambda'])**2 - x['lambda']) / x['lambda'], axis=1)
    aux_olsr_results = sm.OLS(result['AUX_OLS_DEP'], result['lambda']).fit()
    alpha = aux_olsr_results.params
    alpha_se = aux_olsr_results.bse

    return alpha[0], alpha_se[0]

def fitNB(data, features, y_name='Obs'):
    '''
    Fits a NB model to the given data with feature names and target name.
    Returns fitted NB model
    '''
    
    result = data.copy()
    result.reset_index(inplace=True, drop=True)
    X = sm.add_constant(data[features])
    y =  data[y_name]

    disperssion, disperssion_se = computeDisperssion(data, features, y_name)

    NB_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha = disperssion)).fit()
    y_pred_NB = pd.DataFrame(NB_model.predict(X))
    return NB_model, disperssion

def predNB(NB_model, data, features, y_name='Obs'):
    '''
    Fits a NB model to the given data with feature names and target name.
    Returns predictions.
    '''
    
    result = data.copy()
    result.reset_index(inplace=True, drop=True)
    X = sm.add_constant(data[features])
    y =  data[y_name]

    y_pred_NB = pd.DataFrame(NB_model.predict(X))
    return y_pred_NB

def MAPE(actual, predicted):
    '''
    Returns MAPE error.
    '''
    res = np.empty(len(actual))
    for j in range(len(actual)):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return np.mean(np.abs(res))

def forward_regression(data, features, y_name='Obs', threshold_in=0.05, verbose=False):
    '''
    Returns the significant features in a NB forward-stepwise regresion
    '''

    X = data[features]
    y =  data[y_name]

    initial_list = []
    included = []
    while True:
        changed=False
        excluded = list(set(features)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:

            model, alpha = fitNB(data, included+[new_column], y_name)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(data, features, y_name='Obs', threshold_out=0.05, verbose=False):
    '''
    Returns the significant features in a NB backward-stepwise regresion
    '''

    X = data[features]
    y =  data[y_name]

    included=list(X.columns)
    while True:
        changed=False
        model, alpha = fitNB(data, included, y_name)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def NBEBranking(CrashData, alpha, y_name='Obs'):

    CrashData['w'] = 1/(1+alpha*CrashData['NB_pred']) 
    CrashData['EB'] = CrashData['w'] * CrashData['NB_pred']  + (1-CrashData['w']) * CrashData[y_name]
    CrashData['rank'] = CrashData['EB'].rank(pct=True)

    return CrashData['rank']

def NBPSIranking(CrashData, alpha, y_name='Obs'):

    CrashData['w'] = 1/(1+alpha*CrashData['NB_pred']) 
    CrashData['EB'] = CrashData['w'] * CrashData['NB_pred']  + (1-CrashData['w']) * CrashData[y_name]
    CrashData['PSI'] = CrashData['EB'] - CrashData['NB_pred']
    CrashData.loc[CrashData['PSI']<0, 'PSI'] = 0
    CrashData['rank'] = CrashData['PSI'].rank(pct=True)

    return CrashData['rank']

def FI(results):
    FI_test = 0
    for HS_level in [0.02,0.04,0.06,0.08,0.1]:
        FI_test += 1-results[(results['true_rank'] > 1- HS_level) & (results['rank'] > 1- HS_level)].count()[0]/results[(results['rank'] > 1- HS_level)].count()[0]
    return round(FI_test/5, 3)

def PMD(results):
    PMD_test = 0
    for HS_level in [0.02,0.04,0.06,0.08,0.1]:
        PMD_test += (results[results['true_rank'] > 1-HS_level]['Lambda'].mean() - results[results['rank'] > 1-HS_level]['Lambda'].mean())/results[results['true_rank'] > 1-HS_level]['Lambda'].mean()
    return round(PMD_test/5,3)

def CURE(data, y_name= 'Obs', ftot='Ftot', y_pred = 'y_pred'):
    data.sort_values(by=ftot, ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['res'] = data[y_pred] - data[y_name]
    data['res_sq'] = (data[y_pred] - data[y_name])**2

    data['e1'] = data['res'].cumsum()
    data['e2'] = data['res_sq'].cumsum()
    data['e3'] = data['e2']*((1 - data['e2']/data['e2'][len(data['e2'])-1]))

    plt.plot(data[ftot], data['e3'], linestyle = 'dotted', label = "2$\sigma$")
    plt.plot(data[ftot], -data['e3'], linestyle = 'dotted', label = "-2$\sigma$")
    plt.plot(data[ftot], data['e1'], label = "Cum. Residuals")
    plt.legend(loc='lower right')
    plt.xlabel('Total AADT')
    plt.ylabel('CURE')
    plt.show()
