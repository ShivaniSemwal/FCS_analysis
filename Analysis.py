# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:59:48 2023

@author: shivani
This script is to analyse the FCS data.
It plots correlation data (G(T)) vs time (T(s)) and fits the curve 
ToDo: analyse multiple data files and automate the analysis process
"""
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as gridspec
from scipy.optimize import curve_fit # for curve fitting
import tkinter as tk
from tkinter import filedialog
#del file_path

#load files: multiple files can be selected

file_paths = filedialog.askopenfilenames(
        title="Select Data Files",
        filetypes=[("Text files", "*.asc")]
    )
# store multiple data
data_list = []

#read & store data from different files
for file_path in file_paths:
        df = pd.read_fwf(file_path, skiprows=28)
        sz = len(df)
        df.columns = ['a','b','c','d','e']
        df = df.drop(['d','e'], axis = 1)
        idx1 = df.index[df['a'].str.contains('Corr') == True]
        idx2 = df.index[df['a'].str.contains('Count') == True]
        Correlation = df.iloc[ idx1[0]+1: idx2[0]-1].reset_index(drop=True)
        countrate = df.iloc[idx2[0]+1 : sz-2].reset_index(drop=True)
        FCSdata = pd.concat([Correlation,countrate], axis = 1)
        FCSdata.columns = ['time','corr','corr2','countrate','intensity1','intensity2']
        FCSdata = FCSdata.dropna()
        data_list.append(FCSdata)

fig, ax = plt.subplots()
#plt.plot(x,y)
# plot all the data in one plot
for FCSdata in data_list:
    x_data = FCSdata['time'].astype(float)
    y_data = FCSdata['corr'].astype(float)
    plt.semilogx(x_data, y_data,marker='o')
    #plt.semilogx(FCSdata['time'].astype(float), FCSdata['corr'].astype(float),marker='o')
i = 2    # i = 1 for single component fit otherwise multiple component fit
if i==1:
    init_vals = [0.5,0.38,0.0009,.12] # initial guess diff2
    def func(tau,N,T,tau_T,tau_D):
        Rsqd = 25
        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
else:
    init_vals = [5,0.35,0.002,0.99,0.28,1] # initial guess
    def func(tau,N,T,tau_T,f,tau_D1,tau_D2):
        Rsqd = 25
        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

# fit and plot data with curvefit data and fitting data on the plot
        #To do: fitting data text for multiple data should not overlap: fix it 
for FCSdata in data_list:
    x = FCSdata['time'].astype(float)
    y = FCSdata['corr'].astype(float)
    y = y.dropna()
    x = x[:len(y)]
    best_vals, covar = curve_fit(func, x, y, p0=init_vals)
    if i == 1:
        N,T,tau_T,tau_D = best_vals
        errN, errT,errtau_T,errtau_D = np.sqrt(np.diag(covar))
        textstr = '\n'.join((
                r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                r'$\tau_D=%.4f$ $\pm$ %.4f' % (tau_D, errtau_D))) 
    else:
        N,T,tau_T,f,tau_D1,tau_D2 = best_vals
        errN,errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
        textstr = '\n'.join((
                r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
                r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
    #plot curvefit on the data    
    plt.plot(x, func(x, *best_vals), color='black', markersize=8, linestyle='-') 
    
    plt.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10)
    
    
#    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top')

plt.xlabel('T (s)', fontsize=16)
plt.ylabel('g(T)', fontsize=16)


# Display the plot
plt.show()
