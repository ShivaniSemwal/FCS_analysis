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
import math as mt
import csv

" load files"
filelist=glob.glob('D:/FCS_analysis/Data/*.asc')
filelist.sort()
filelist

"read data"
df = pd.read_fwf(filelist[0], skiprows=28)
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
x = FCSdata['time'].astype(float)
y = FCSdata['corr'].astype(float)
fig = plt.figure(figsize=(15, 12),dpi=80)
plt.plot(x,y)
plt.xscale('log')
init_vals = [40,25,0.2,0.0002,0.00052]

def func(tau,N,Rsqd,T,tau_T,tau_D):
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

x = FCSdata['time'].astype(float)
y = FCSdata['corr'].astype(float)
y = y.dropna()
x = x[:len(y)]
best_vals, covar = curve_fit(func, x, y, p0=init_vals)
print('best_vals: {}'.format(best_vals))
plt.plot(x, func(x, *best_vals), 'r-')
plt.xscale('log')
plt.xlabel('T (s)', fontsize=16)
plt.ylabel('g(T)', fontsize=16)
plt.show()
fig.savefig('FCS.png')