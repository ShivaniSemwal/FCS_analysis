# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:41:56 2023

@author: shiva
"""

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig
import pandas as pd

# Create a Tkinter root window (it will not be displayed)
root = tk.Tk()
root.withdraw()

# Initialize an empty list to store selected file paths
i = 1

if i == 1:
    file_paths = []
    while True:
        file_path = filedialog.askopenfilename(
                title="Select a File",
                filetypes=[("Text files", "*.asc")],
                initialdir="/path/to/your/folder"  # Optional: specify a starting directory
                )
    
        if not file_path:
            break  # User canceled the selection
        else:
            file_paths.append(file_path)

# Check if files were selected
    if file_paths:
        print("Selected files:")
        for file_path in file_paths:
            print(file_path)
    else:
        print("No files selected.")

else:
    file_paths = filedialog.askopenfilenames(filetypes=[("ASC Files", "*.asc")])
        
def diff2(tau,N,T,tau_T,tau_D):
    Rsqd = 25
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

def diff3(tau,N,T,tau_T,f,tau_D1,tau_D2):
    Rsqd = 25
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

initial_params = {
    "diff2minus1": {'N': 4.0, 'T': 0.38, 'tau_T': 0.0088,'tau_D': 0.2},
    "diff3minus1": {'N': 5.0, 'T': 0.35, 'tau_T': 0.002, 'f': 0.99,'tau_D1': 0.28, 'tau_D2': 1}
}

    
    
# You can now use the selected file paths as needed.
def update_graph():
    selected_model = model_var.get()
    params = [float(param_entry.get()) for param_entry in param_entries[selected_model]]   
    if selected_model == "diff2minus1":
        best_vals, covar = curve_fit(diff2, x_data, y_data, params)
        fitted_data = diff2(x_data, *best_vals)
    elif selected_model == "diff3minus1":
        best_vals, covar = curve_fit(diff3 ,x_data, y_data, params)
        fitted_data = diff3(x_data, *best_vals)
    else:
        return  # No valid model selected
    
    ax.clear()
    
    ax.plot(x_data, y_data, label='Data', color='blue')
    ax.plot(x_data, fitted_data, label=f'Fitted {selected_model}', color='red')
    ax.legend()
    # Change the axis limits based on user input
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    y_min = np.min(y_data)
    y_max = np.max(y_data)
    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('T(s)')
    ax.set_ylabel('G(T)')
    
    canvas.draw()

def load_and_plot_data():
    file_paths = filedialog.askopenfilenames(
        title="Select Data Files",
        filetypes=[("Text files", "*.asc")]
    )
    
    if not file_paths:
        return  # No files selected
    
    global x_data, y_data
    x_data, y_data = [], []
    
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
        x_data = FCSdata['time'].astype(float)
        y_data = FCSdata['corr'].astype(float)
    
    update_graph()
root = tk.Tk()
root.title("Fit Data with Fitting Functions")

# Create a label and dropdown to select the fitting function
model_label = ttk.Label(root, text="Select Fitting Function:")
model_label.pack()    

models = ["diff2minus1", "diff3minus1"]  # Add more fitting functions if needed
model_var = tk.StringVar(value=models[0])
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=models)
model_dropdown.pack()

# Create input fields for initial parameters
param_entries = {}
for model in models:
    param_frame = ttk.Frame(root)
    param_frame.pack()
    ttk.Label(param_frame, text=f"Initial Parameters for {model}:").pack()
    param_entries[model] = []
    for param_name in initial_params[model]:
        ttk.Label(param_frame, text=param_name).pack(side=tk.LEFT)
        param_entry = ttk.Entry(param_frame)
        param_entry.insert(0, str(initial_params[model][param_name]))
        param_entry.pack(side=tk.LEFT)
        param_entries[model].append(param_entry)

# Create a button to fit the selected model to the data and update the graph
fit_button = ttk.Button(root, text="Fit Data & Update Graph", command=update_graph)
fit_button.pack()

# Create a button to load data from files and plot it
load_button = ttk.Button(root, text="Load Data Files", command=load_and_plot_data)
load_button.pack()

# Create a matplotlib figure and canvas to display the graph
mpl_fig.Figure(figsize=(6, 4), dpi=100)
fig = plt.Figure()
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Update the graph initially
update_graph()

# Start the tkinter main loop
root.quit()
#fig, ax = plt.subplots()
#for file in file_paths:
#
#    df = pd.read_fwf(file, skiprows=28)
#    sz = len(df)
#    df.columns = ['a','b','c','d','e']
#    df = df.drop(['d','e'], axis = 1)
#    idx1 = df.index[df['a'].str.contains('Corr') == True]
#    idx2 = df.index[df['a'].str.contains('Count') == True]
#    Correlation = df.iloc[ idx1[0]+1: idx2[0]-1].reset_index(drop=True)
#    countrate = df.iloc[idx2[0]+1 : sz-2].reset_index(drop=True)
#    FCSdata = pd.concat([Correlation,countrate], axis = 1)
#    FCSdata.columns = ['time','corr','corr2','countrate','intensity1','intensity2']
#    FCSdata = FCSdata.dropna()
#    x = FCSdata['time'].astype(float)
#    y = FCSdata['corr'].astype(float)
##fig= plt.figure(figsize=(15, 12),dpi=80)
#    
#    plt.plot(x,y)
#    plt.xscale('log')
#    
#
#   # global_model = Model(func)
#    
#    
#    x = FCSdata['time'].astype(float)
#    y = FCSdata['corr'].astype(float)
#    y = y.dropna()
#    x = x[:len(y)]
##best_vals, covar = curve_fit(glob_fun, x, y.ravel(),p0=init_vals)
#    best_vals, covar = curve_fit(func, x, y, p0=init_vals)
#    print('covar: {}'.format(covar))
#    print('best_vals: {}'.format(best_vals))
##N,Rsqd,tau_D = best_vals
##N,T,tau_T,tau_D = best_vals
#    N,T,tau_T,f,tau_D1,tau_D2 = best_vals
#    errN,errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
#    plt.plot(x, func(x, *best_vals), 'r-')
#    plt.xscale('log')
#    plt.xlabel('T (s)', fontsize=16)
#    plt.ylabel('g(T)', fontsize=16)
#    textstr = '\n'.join((
#            r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
#            #  r'$Rsqd=%.2f$' % (Rsqd, ),
#            r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
#            r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
#            #  r'$\tau_D=%.4f$ $\pm$ %.4f' % (tau_D, )))
#            r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
#            r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
#            r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
#    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top')
#    plt.show()


