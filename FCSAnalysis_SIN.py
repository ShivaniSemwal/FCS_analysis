# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:37:46 2023

@author: shivani
This script is to analyse the FCS data.
It plots correlation data (G(T)) vs time (T(s)) and fits the curve for multiple files
Saves the final analysed data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # for curve fitting
from tkinter import filedialog
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.simpledialog import askstring#del file_path
import csv


#load files: multiple files can be selected
root = tk.Tk()
root.withdraw()

# select files if not already selected from a folder

#if 'file_paths' in locals():
#    print("File path is defined.")
#else:
#    file_paths = filedialog.askopenfilenames(
#            title="Select Data Files",
#            filetypes=[("Text files", "*.SIN")]
#            )


file_paths = filedialog.askopenfilenames(
            title="Select Data Files",
            filetypes=[("Text files", "*.SIN")]
            )    
# store data from different data files
data_list = []
countrate_list = []
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
        dk = pd.read_fwf(file_path)
        dk.columns = ['a','b']
        idx_1 = dk.index[dk['a'].str.contains('Trace') == True]
        skip_row = idx_1[0]+1
        dk = pd.read_fwf(file_path, skiprows=skip_row)
        dk.columns = ['a']
        idx_2 = dk.index[dk['a'].str.contains('Histogram') == True]
        dk = dk.iloc[0:idx_2[0]-2]
        dk['a'] = dk['a'].str.replace(r'\s+', ' ', regex=True)
        dk['a'] = dk['a'].str.split(' ')
        dk[['col1', 'detector1', 'detector2']] = pd.DataFrame(dk['a'].tolist(), dtype=float)
        avg_intensity = dk['detector1'].mean()/1000 + dk['detector2'].mean()/1000  #KHz
        countrate_list.append(avg_intensity)
        

fig, ax = plt.subplots()
# start and end index of the data to be plotted
a = 70  
b = 450
#plt.plot(x,y)

# plot all the data in one plot
for index, FCSdata in enumerate(data_list):    
    x_data = FCSdata['time'].astype(float)
    y_data = FCSdata['corr1'].astype(float)-1
    #ToDo: remove file location 'D/' from label in the plot
    plt.semilogx(x_data[a:b], y_data[a:b], marker='o',label = file_paths[index]) 
    plt.legend(loc="upper right")
    #plt.semilogx(FCSdata['time'].astype(float), FCSdata['corr1'].astype(float)-1,marker='o')
  


# select variable 'i' based on the type of fit you want    
parameter_value = askstring("i-value", "Enter i = (1: single_comp_fit,   2: hold tau_T_single_comp_fit,  3: double_compfit,  4: hold f_double_compfit,  5: hold tau_D1_double_compfit, 6: hold tau_T_double_compfit")
# Check the response
if parameter_value:
    i = int(parameter_value)
    print("User entered parameter value:", parameter_value)
    if i==1:
        # set the initial guess param
        init_vals = [164,0.15,0.000006,.000039] # initial guess for atto_100nm_10kda
        #init_vals = [150,0.17,0.00006,.0001]
        def func(tau,N,T,tau_T,tau_D):  #single comp fit
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

    elif i == 2:
        init_vals = [164,0.1,.0001] # initial guess diff2
        holdTau_t = askstring("Tau_T-value", "Enter Tau_T value")
        tau_T = float(holdTau_t)
        def func(tau,N,T,tau_D): #single comp fit and hold tau_t
            #tau_T = 0.000006
            #tau_T = 0.0000009
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

    elif i == 3:
        #init_vals = [76,0.1,0.000001,0.1,0.000002,0.01] # initial guess
        init_vals = [15,0.12,0.000005,0.4,0.00006,0.0004]
        def func(tau,N,T,tau_T,f,tau_D1,tau_D2): #double comp fit
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
    elif i == 4:
        init_vals = [76,0.89,0.000001,0.002,0.1] # initial guess
        hold_f = askstring("f-value", "Enter f value")
        f = 0.28
        def func(tau,N,T,tau_T,tau_D1,tau_D2): #double comp fit and hold f
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

    elif i == 5:
        init_vals = [76,0.1,0.000001,0.1,0.01] # initial guess
        hold_tau_D1 = askstring("tau_D1-value", "Enter tau_D1 value")
        tau_D1 = float(hold_tau_D1)
        def func(tau,N,T,tau_T,f,tau_D2): #double comp fit and hold tau_D1
            
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

    elif i == 6:
        init_vals = [76,0.1,0.000001,0.1,0.01] # initial guess
        hold_tau_T = askstring("tau_T-value", "Enter tau_T value")
        tau_T = float(hold_tau_T)
        def func(tau,N,T,f,tau_D1,tau_D2): #double comp fit and hold tau_D1         
            Rsqd = 25
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
   

# fit and plot data with curvefit data and fitting data on the plot
    best_val_list = []   
    
    for index, FCSdata in enumerate(data_list):  
        x = FCSdata['time'].astype(float)
        y = FCSdata['corr1'].astype(float)-1
    #y = y.dropna()
    #x = x[:len(y)]
        best_vals, covar = curve_fit(func, x[a:b], y[a:b], p0=init_vals)
        best_val_list.append(best_vals)        
        print(best_vals)   
        plt.plot(x[a:b], func(x[a:b], *best_vals), color='black', markersize=8, linestyle='-')
    size = best_vals.shape[0]
    mat = np.array(best_val_list, dtype=float).reshape(index+1, size)
    
    #function for inputting concentration values
    def insert_conc():
        root = tk.Tk()
        root.withdraw()  

        # Show an input dialog to enter concentration (comma-separated)
        input_str = simpledialog.askstring("Insert concentration", "Enter concentration (comma-separated):")

        # Close the hidden root window
        root.destroy()

        # Process the input and convert it to a list
        if input_str:
            conc_list = input_str.split(',')
            conc_list = [item.strip() for item in conc_list]  # Remove leading/trailing spaces
            return conc_list
        else:
            return None
    conc = insert_conc()
    if i == 1 or i == 2:    
        col_name = np.array(['conc', 'avg_countrate (KHz)', 'N','T','tau_T','tau_D'])
    else:
        col_name = np.array(['conc', 'avg_countrate (KHz)', 'N','T','tau_T','f','tau_D1','tau_D2'])
    all_data = np.insert(mat, 0, countrate_list, axis=1)
    FCS_data = np.insert(all_data, 0, conc, axis=1   ) 
    final_FCS_data = np.vstack((col_name, FCS_data)) 


   
#    for FCSdata in data_list:
#        x = FCSdata['time'].astype(float)
#        y = FCSdata['corr1'].astype(float)-1
#    #y = y.dropna()
#    #x = x[:len(y)]
#        best_vals, covar = curve_fit(func, x[a:b], y[a:b], p0=init_vals)
#        best_val_list.append(best_vals)
#        best_val_list.append(countrate_list[i])
#        print(best_vals)   
#        plt.plot(x[a:b], func(x[a:b], *best_vals), color='black', markersize=8, linestyle='-')
        
# display parameter values in the plot
# ToDo: nicely arrange all the param values from multiple data fit in the plot
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
            #tau_T = 0.000009
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
                    r'$T=%.4f$ $\pm$ %.4f' % (T, errT ),
                    r'$\tau_T=%.8f$ $\pm$ %.8f' % (tau_T, errtau_T ),
                    r'$f=%.2f$ $\pm$ %.2f' % (f, errf ),
                    r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                    r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
        elif i == 4:
           # f = 0.28
            N,T,tau_T,tau_D1,tau_D2 = best_vals
            errN,errT,errtau_T,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
            textstr = '\n'.join((
                    r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                    r'$\tau_T=%.8f$ $\pm$ %.8f' % (tau_T, errtau_T ),
                    r'$f=%.2f$' % (f),
                    r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                    r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
        
        elif i == 5:
            #tau_D1 = 0.00006
            N,T,tau_T,f,tau_D2 = best_vals
            errN,errT,errtau_T,errf,errtau_D2 = np.sqrt(np.diag(covar))      
            textstr = '\n'.join((
                    r'$N=%.6f$ $\pm$ %.6f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.6f$ $\pm$ %.6f' % (T, errT ),
                    r'$f=%.8f$ $\pm$ %.8f' % (f, errf ),
                    r'$\tau_D1=%.6f$' % (tau_D1 ),
                     r'$\tau_T=%.6f$ $\pm$ %.6f' % (tau_T, errtau_T ),
                    r'$\tau_D2=%.6f$ $\pm$ %.6f' % (tau_D2, errtau_D2)))
        elif i == 6:
            #tau_D1 = 0.00006
            N,T,f,tau_D1,tau_D2 = best_vals
            errN,errT,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))      
            textstr = '\n'.join((
                    r'$N=%.6f$ $\pm$ %.6f' % (N, errN),
                    # r'$Rsqd=%.2f$ $\pm$ %.4f' % (Rsqd, errR ),
                    r'$Rsqd=%.2f$ ' % (Rsqd ),
                    r'$T=%.6f$ $\pm$ %.4f' % (T, errT ),
                    r'$f=%.2f$ $\pm$ %.2f' % (f, errf ),
                     r'$\tau_T=%.8f$ ' % (tau_T ),
                    r'$\tau_D1=%.6f$ $\pm$ %.6f' % (tau_D1,errtau_D1 ),                    
                    r'$\tau_D2=%.6f$ $\pm$ %.6f' % (tau_D2, errtau_D2)))
        
        # x and y position of param values in the plot 
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
    
#Save Data    
file_name = 'final_FCS_data.csv'                 
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)

    for row in final_FCS_data:
       writer.writerow(row)
root.quit()