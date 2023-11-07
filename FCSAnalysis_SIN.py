# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:37:46 2023

@author: shivani
This script is to analyse the FCS data.
It plots correlation data (G(T)) vs time (T(s)) and fits the curve 
ToDo: analyse multiple data files and automate the analysis process
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # for curve fitting
from tkinter import filedialog
import tkinter as tk
from tkinter import messagebox
from tkinter.simpledialog import askstring#del file_path

#load files: multiple files can be selected
root = tk.Tk()
root.withdraw()
if 'file_paths' in locals() and file_path:
    print("File path is defined.")
else:
    file_paths = filedialog.askopenfilenames(
            title="Select Data Files",
            filetypes=[("Text files", "*.SIN")]
            )
# store multiple data
data_list = []
file = []
#read & store data from different files
for file_path in file_paths:
        df = pd.read_fwf(file_path, skiprows=16)
        sz = len(df)
        df.columns = ['a','b','c']
        idx1 = df.index[df['a'].str.contains('Corr') == True]
#        idx2 = df.index[df['a'].str.contains('Count') == True]
#        Correlation = df.iloc[ idx1[0]+1: idx2[0]-1].reset_index(drop=True)
#        countrate = df.iloc[idx2[0]+1 : sz-2].reset_index(drop=True)
        FCSdata = df.iloc[0:idx1[0]-1]
        FCSdata.columns = ['time','corr1','corr2']
        FCSdata = FCSdata.replace(0, np.nan)
        FCSdata =  FCSdata.dropna()
        #FCSdata = FCSdata.dropna()      
        #fig, ax = plt.subplots()
        #FCSdata = FCSdata.dropna()
        data_list.append(FCSdata)
        file.append(file_path)

fig, ax = plt.subplots()
a = 70
b = 450
#plt.plot(x,y)
# plot all the data in one plot
for index, FCSdata in enumerate(data_list):
    
    x_data = FCSdata['time'].astype(float)
    y_data = FCSdata['corr1'].astype(float)-1
    plt.semilogx(x_data[a:b], y_data[a:b], marker='o',label = file_paths[index])
    plt.legend(loc="upper right")
    #plt.semilogx(FCSdata['time'].astype(float), FCSdata['corr1'].astype(float)-1,marker='o')
  


# Display an 'askyesno' dialog box
parameter_value = askstring("i-value", "Enter i = (1: single_comp_fit,   2: hold tau_T_single_comp_fit,  3: double_compfit,  4: hold f_double_compfit,  5: hold tau_D1_double_compfit")
# Check the response
if parameter_value:
    i = int(parameter_value)
    print("User entered parameter value:", parameter_value)
    if i==1:
        init_vals = [164,0.15,0.000006,.000039] # initial guess for atto_100nm_10kda
        #init_vals = [150,0.17,0.00006,.0001]
        def func(tau,N,T,tau_T,tau_D):
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

    elif i == 2:
        init_vals = [164,0.1,.0001] # initial guess diff2
        def func(tau,N,T,tau_D):
            tau_T = 0.000006
            #tau_T = 0.0000009
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

    elif i == 3:
        #init_vals = [76,0.1,0.000001,0.1,0.000002,0.01] # initial guess
        init_vals = [15,0.12,0.000005,0.4,0.00006,0.0004]
        def func(tau,N,T,tau_T,f,tau_D1,tau_D2):
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
    elif i == 4:
        init_vals = [76,0.89,0.000001,0.002,0.1] # initial guess
        def func(tau,N,T,tau_T,tau_D1,tau_D2):
            f = 0.28
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

    elif i == 5:
        init_vals = [76,0.1,0.000001,0.1,0.01] # initial guess
        def func(tau,N,T,tau_T,f,tau_D2):
            tau_D1 = 0.000006
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))


   
#i = 1   # i = 1 for single component fit otherwise multiple component fit
# fit and plot data with curvefit data and fitting data on the plot
        #To do: fitting data text for multiple data should not overlap: fix it 
    best_val_list = []      
  
    for FCSdata in data_list:
        x = FCSdata['time'].astype(float)
        y = FCSdata['corr1'].astype(float)-1
    #y = y.dropna()
    #x = x[:len(y)]
        best_vals, covar = curve_fit(func, x[a:b], y[a:b], p0=init_vals)
        best_val_list.append(best_vals)
        print(best_vals)
    
        plt.plot(x[a:b], func(x[a:b], *best_vals), color='black', markersize=8, linestyle='-')
        

    for index, best_vals in enumerate(best_val_list): 
        Rsqd = 25
        if i == 1:
            N,T,tau_T,tau_D = best_vals
            errN, errT,errtau_T,errtau_D = np.sqrt(np.diag(covar))        
            textstr = '\n'.join((
                    r'$N=%.6f$ $\pm$ %.6f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.6f$ $\pm$ %.4f' % (T, errT ),
                    #r'$\tau_T=%.6f$' % (tau_T ),
                    r'$\tau_T=%.6f$ $\pm$ %.6f' % (tau_T, errtau_T ),
                    r'$\tau_D=%.6f$ $\pm$ %.6f' % (tau_D, errtau_D))) 
        elif i == 2:
            tau_T = 0.000009
        #tau_T = 0.000009
            N,T,tau_D = best_vals
            errN, errT,errtau_D = np.sqrt(np.diag(covar))      
            textstr = '\n'.join((
                    r'$N=%.6f$ $\pm$ %.6f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.6f$ $\pm$ %.4f' % (T, errT ),
                    r'$\tau_T=%.6f$' % (tau_T ),
                    # r'$\tau_T=%.6f$ $\pm$ %.6f' % (tau_T, errtau_T ),
                    r'$\tau_D=%.6f$ $\pm$ %.6f' % (tau_D, errtau_D)))
        elif i == 3:
        
            N,T,tau_T,f,tau_D1,tau_D2 = best_vals
            errN,errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
            textstr = '\n'.join((
                    r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                    #r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                    r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                    r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
                    r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                    r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
        elif i == 4:
            f = 0.28
            N,T,tau_T,tau_D1,tau_D2 = best_vals
            errN,errT,errtau_T,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
            textstr = '\n'.join((
                    r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                    r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                    r'$f=%.2f$' % (f),
                    r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                    r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
        
        elif i == 5:
            tau_D1 = 0.00006
            N,T,tau_T,f,tau_D2 = best_vals
            errN,errT,errtau_T,errf,errtau_D2 = np.sqrt(np.diag(covar))      
            textstr = '\n'.join((
                    r'$N=%.6f$ $\pm$ %.6f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.6f$ $\pm$ %.4f' % (T, errT ),
                    r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
                    r'$\tau_D1=%.6f$' % (tau_D1 ),
                    # r'$\tau_T=%.6f$ $\pm$ %.6f' % (tau_T, errtau_T ),
                    r'$\tau_D2=%.6f$ $\pm$ %.6f' % (tau_D2, errtau_D2)))
        
        #plot curvefit on the data    
        if index == 0:
            plt.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=10)
        else:
            if i == 1 or i == 2:
                plt.text(0.45, 0.95-index+0.85*index, textstr, transform=ax.transAxes, fontsize=10)
            else:
                plt.text(0.45, 0.95-index+0.8*index, textstr, transform=ax.transAxes, fontsize=10)
        
    
    
#    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top')

    plt.xlabel('T (s)', fontsize=16)
    plt.ylabel('g(T)', fontsize=16)


# Display the plot
    plt.show()
else:
    print("No parameter value provided.")
# Run the tkinter main loop (necessary for the dialog box to appear)
root.quit()