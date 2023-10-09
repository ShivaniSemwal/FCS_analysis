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
" load files"
#filelist=glob.glob('D:/FCS_analysis/Data/*.asc')
#filelist.sort()
#filelist
file_path = filedialog.askopenfilename(filetypes=[("ASC Files", "*.asc")])
"read data"
#df = pd.read_fwf(filelist[3], skiprows=28)
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
x = FCSdata['time'].astype(float)
y = FCSdata['corr'].astype(float)
#fig= plt.figure(figsize=(15, 12),dpi=80)
fig, ax = plt.subplots()
plt.plot(x,y)
plt.xscale('log')
#init_vals = [4,25,0.38] #diff1
#init_vals = [4,0.38,0.0088,0.2] #diff2
init_vals = [5,0.35,0.002,0.99,0.28,1] #diff3, 30:180
#init_vals = [4,25,0.38,0.5,0.2] #diffD1D2

#def func(tau,N,Rsqd,tau_D):
#    return (1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

#def func(tau,N,T,tau_T,tau_D):
#    Rsqd = 25
#    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

def func(tau,N,T,tau_T,f,tau_D1,tau_D2):
    Rsqd = 25
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

def glob_fun(tau,N,T,tau_T,f,tau_D1,tau_D2):
    return np.tile(func(tau,N,T,tau_T,f,tau_D1,tau_D2), len(x))

#def func(tau,N,Rsqd,f,tau_D1,tau_D2):
#    return  (1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5) + (1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

x = FCSdata['time'].astype(float)
y = FCSdata['corr'].astype(float)
y = y.dropna()
x = x[:len(y)]
#best_vals, covar = curve_fit(glob_fun, x, y.ravel(),p0=init_vals)
best_vals, covar = curve_fit(func, x, y, p0=init_vals)
print('covar: {}'.format(covar))
print('best_vals: {}'.format(best_vals))
#N,Rsqd,tau_D = best_vals
#N,T,tau_T,tau_D = best_vals
N,T,tau_T,f,tau_D1,tau_D2 = best_vals
errN,errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
plt.plot(x, func(x, *best_vals), 'r-')
plt.xscale('log')
plt.xlabel('T (s)', fontsize=16)
plt.ylabel('g(T)', fontsize=16)
textstr = '\n'.join((
    r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
  #  r'$Rsqd=%.2f$' % (Rsqd, ),
    r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
    r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
  #  r'$\tau_D=%.4f$ $\pm$ %.4f' % (tau_D, )))
    r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
    r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
    r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
plt.show()
#fig.savefig('FCS.png')