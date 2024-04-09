# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script for analysing FCS data saved in .ASC and .SIN format.
Single and multiple dataset can be loaded using Load Data.
Data can be viewed by plotting it using Plot Graphs.
For data analysis, select fitting model, set initial guess parameters and curve fit.
Save results to a single file using Save file.

Important: 1. The script can hold/fix one fitting parameter at a time
           2. To hold/fix multiple parameters together, the script has been updated to
           hold only tau_T and tau_D1 together.
    ***if needed to hold some other parameters together, 
    script can be modified accordingly***        
"""

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit # for curve fitting
import csv
#from matplotlib.animation import FuncAnimation


class FCSFitting_app:
    def __init__(self, root):
        self.root = root
        self.root.title("FCS analysis")

        # Create widgets
        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=0, columnspan=2, pady=10)
        
        
        self.load_button = tk.Button(root, text="curve fit", command=self.update_graph)
        self.load_button.grid(row=0, column=5, columnspan=7, pady=10)
        
        self.load_button = tk.Button(root, text="save file", command=self.save)
        self.load_button.grid(row=10, column=1, columnspan=1, pady=10)
        
        self.load_button = tk.Button(root, text="plot graph", command=self.plot_graph)
        self.load_button.grid(row=0, column=3, columnspan=1, pady=10)
        
        self.load_button = tk.Button(root, text="clear loaded data", command=self.clear_loaded_data)
        self.load_button.grid(row=10, column=3, columnspan=1, pady=10)

#        self.text_widget = tk.Text(root, height=10, width=50)
#        self.text_widget.pack(pady=10)
        
        self.data_list = []
        self.countrate_list = []
        self.file = []
        self.start_end_point = []
        self.best_val_list = [] 
        self.error = []
        self.data = 0
        self.size = 0
        self.mat = 0
        self.errmat = 0
        self.ind = 0
         # Entry widgets for initial parameters
       
        self.entry_a = tk.Entry(root)
        self.entry_b = tk.Entry(root)
        self.entry_c = tk.Entry(root)
        self.entry_d = tk.Entry(root)
        self.entry_e = tk.Entry(root)
        self.entry_f = tk.Entry(root)
        self.entry_g = tk.Entry(root)
        
        self.entry_conc = tk.Entry(root)
        self.entry_file = tk.Entry(root)
        
        # Checkbuttons for parameters
        self.var_a = tk.IntVar()
        self.var_b = tk.IntVar()
        self.var_c = tk.IntVar()
        self.var_d = tk.IntVar()
        self.var_e = tk.IntVar()
        self.var_f = tk.IntVar()
        self.var_g = tk.IntVar()
        

        # Increase the font size of the Checkbuttons
        checkbutton_font = ('Helvetica', 15)  # Adjust the font size as needed

        self.check_a = tk.Checkbutton(root, variable=self.var_a, font=checkbutton_font)
        self.check_a.grid(row=3, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed

        self.label_a = tk.Label(root, text='N:')
        self.label_a.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.entry_a.grid(row=3, column=2, pady=2)
        self.entry_a.insert(1, '160.0') 

        self.check_b = tk.Checkbutton(root, variable=self.var_b, font=checkbutton_font)
        self.check_b.grid(row=4, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed

        self.label_b = tk.Label(root, text='T:')
        self.label_b.grid(row=4, column=1, sticky=tk.W, pady=2)
        self.entry_b.grid(row=4, column=2, pady=2)
        self.entry_b.insert(0, '0.12') 

        self.check_c = tk.Checkbutton(root, variable=self.var_c, font=checkbutton_font)
        self.check_c.grid(row=5, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed
        

        self.label_c = tk.Label(root, text='tau_T:')
        self.label_c.grid(row=5, column=1, sticky=tk.W, pady=2)
        self.entry_c.grid(row=5, column=2, pady=2)
        self.entry_c.insert(0, '0.00002') 
        
        self.check_d = tk.Checkbutton(root, variable=self.var_d, font=checkbutton_font)
        self.check_d.grid(row=6, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed

        self.label_d = tk.Label(root, text='tau_D:')
        self.label_d.grid(row=6, column=1, sticky=tk.W, pady=2)
        self.entry_d.grid(row=6, column=2, pady=2)
        self.entry_d.insert(0, '0.13') 
        
        self.check_e = tk.Checkbutton(root, variable=self.var_e, font=checkbutton_font)
        self.check_e.grid(row=6, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed
        
        self.label_e = tk.Label(root, text='f:')
        self.label_e.grid(row=6, column=1, sticky=tk.W, pady=2)
        self.entry_e.grid(row=6, column=2, pady=2)
        self.entry_e.insert(0, '0.4') 
        
        self.check_f = tk.Checkbutton(root, variable=self.var_f, font=checkbutton_font)
        self.check_f.grid(row=7, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed
        
        self.label_f = tk.Label(root, text='tau_D1:')
        self.label_f.grid(row=7, column=1, sticky=tk.W, pady=2)
        self.entry_f.grid(row=7, column=2, pady=2)
        self.entry_f.insert(0, '0.00002') 
        
        self.check_g = tk.Checkbutton(root, variable=self.var_g, font=checkbutton_font)
        self.check_g.grid(row=8, column=0, sticky=tk.W, ipadx=5)  # Adjust ipadx as needed
        
        self.label_g = tk.Label(root, text='tau_D2:')
        self.label_g.grid(row=8, column=1, sticky=tk.W, pady=2)
        self.entry_g.grid(row=8, column=2, pady=2)
        self.entry_g.insert(0, '0.00001') 
        
        self.label_conc = tk.Label(root, text='concentration:')
        self.label_conc.grid(row=9, column=1, sticky=tk.W, pady=2)
        self.entry_conc.grid(row=9, column=2, pady=2)
        self.entry_conc.insert(1, '0') 
        
        self.label_file = tk.Label(root)
        self.label_file.grid(row=10, column=1, sticky=tk.W, pady=2)
        self.entry_file.grid(row=10, column=2, pady=2)
        self.entry_file.insert(1, 'file name.csv') 
        
        # Dropdown menu for selecting the fitting model
        self.model_var = tk.StringVar()
        self.model_var.set("Select curve fit model")  # Default selection
        models = ["Single Component Fitting", "Double Component Fitting"]
        self.model_menu = tk.OptionMenu(root, self.model_var, *models)
        self.model_menu.grid(row=0, column=2, padx=10)
        
        # Bind the model change event
        self.model_var.trace_add("write", self.update_model_parameters)
        
    def update_model_parameters(self, *args):
        selected_model = self.model_var.get()

        if selected_model == "Single Component Fitting":
            self.check_e.grid_remove()
            self.label_e.grid_remove()
            self.entry_e.grid_remove()
            self.check_f.grid_remove()
            self.label_f.grid_remove()
            self.entry_f.grid_remove()
            self.check_g.grid_remove()
            self.label_g.grid_remove()
            self.entry_g.grid_remove()

            self.label_a.grid()
            self.entry_a.grid()
            self.label_b.grid()
            self.entry_b.grid()
            self.label_c.grid()
            self.entry_c.grid()
            self.label_d.grid()
            self.entry_d.grid()
            
        elif selected_model == "Double Component Fitting":
            self.label_d.grid_remove()
            self.entry_d.grid_remove()          
            self.label_a.grid()
            self.entry_a.grid()
            self.label_b.grid()
            self.entry_b.grid()
            self.label_c.grid()
            self.entry_c.grid()
            self.check_e.grid()
            self.label_e.grid()
            self.entry_e.grid()
            self.check_f.grid()
            self.label_f.grid()
            self.entry_f.grid()
            self.check_g.grid()
            self.label_g.grid()
            self.entry_g.grid()

        # Update the plot
        #self.update_graph()
    def fix_param(self):
        fix_N = self.var_a.get()
        fix_T = self.var_b.get()
        fix_tau_T = self.var_c.get()
        fix_tau_D = self.var_d.get()
        fix_f = self.var_e.get()
        fix_tau_D1 = self.var_f.get()
        fix_tau_D2 = self.var_g.get()
        
        return fix_N, fix_T, fix_tau_T, fix_tau_D, fix_f, fix_tau_D1, fix_tau_D2
    
    def conc(self):
        input_conc = self.entry_conc.get()
        conc_list = input_conc.split(',')
        conc_list = [item.strip() for item in conc_list]  # Remove leading/trailing spaces
        return conc_list
        
        
    def save(self):   
        conc_list = self.conc()
        print('conc', conc_list)
        if self.model_var.get() == "Single Component Fitting":  
            col_name = np.array(['conc', 'avg_countrate (KHz)', 'N','T','tau_T','tau_D','errN','errT','errtau_T','errtau_D'])
        if self.model_var.get() == "Double Component Fitting":
            col_name = np.array(['conc', 'total_countrate (KHz)', 'N','T','tau_T','f','tau_D1','tau_D2','errN','errT','errtau_T','errf','errtau_D1','errtau_D2'])
        #print('self.best',self.best_val_list)
        self.mat = np.array(self.best_val_list, dtype=float).reshape(self.ind+1, self.size)
        self.errmat = np.array(self.error, dtype=float).reshape(self.ind+1, self.size)
        all_data = np.concatenate((self.mat, self.errmat), axis=1)
        all_data = np.insert(all_data, 0, self.countrate_list, axis=1)
        FCS_data = np.insert(all_data, 0, conc_list, axis=1   ) 
        final_FCS_data = np.vstack((col_name, FCS_data)) 
        print('final data', final_FCS_data)
        file_name = self.entry_file.get()
        print('file_name', file_name)
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in final_FCS_data:
                writer.writerow(row)
        
    def fit_curve(self, x, y, *initial_params, model): 
        print('initial_params',initial_params)
        fix_N, fix_T, fix_tau_T, fix_tau_D, fix_f, fix_tau_D1, fix_tau_D2 = self.fix_param()
        
       # print('param', N, T, tau_T, tau_D, f, tau_D1, tau_D2)
        if model == "Single Component Fitting":
            if fix_N or fix_T or fix_tau_T or fix_tau_D:
                print('hold param:')
                if fix_N:
                    def func(tau,T,tau_T,tau_D):          
                        Rsqd = 25
                        N = float(self.entry_a.get())
                        #del initial_params[0]
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
                    best_vals, covariance = curve_fit(func, x, y, p0=initial_params[1:4])
                    errN = 0
                    errT,errtau_T,errtau_D = np.sqrt(np.diag(covariance))  
                    error = [errN, errT,errtau_T,errtau_D]
                    best_vals = np.insert(best_vals, 0,float(self.entry_a.get()))
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_T:
                    def func(tau,N,tau_T,tau_D):          
                        Rsqd = 25
                        T = float(self.entry_b.get())
                        #T = initial_params[1]
                        #del initial_params[1]
                        #print('T', T)
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[2],initial_params[3]])
                    best_vals = np.insert(best_vals, 1,float(self.entry_b.get()))
                    errT = 0
                    errN,errtau_T,errtau_D = np.sqrt(np.diag(covariance))  
                    error = [errN, errT,errtau_T,errtau_D]
                    #best_vals.insert[1,float(self.entry_b.get())]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_tau_T:
                    def func(tau,N,T,tau_D):          
                        Rsqd = 25
                        tau_T = float(self.entry_c.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[3]])
                    best_vals = np.insert(best_vals, 2,float(self.entry_c.get()))
                    errtau_T = 0
                    errN,errT,errtau_D = np.sqrt(np.diag(covariance))  
                    error = [errN, errT,errtau_T,errtau_D]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals)
                    self.error.append(error)
                    return best_vals
                if fix_tau_D:
                    def func(tau,N,T,tau_T):          
                        Rsqd = 25
                        tau_D = float(self.entry_d.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[2]])
                    errtau_D = 0
                    errN,errT,errtau_T = np.sqrt(np.diag(covariance))
                    error = [errN, errT,errtau_T,errtau_D]
                    best_vals = np.insert(best_vals, 3,float(self.entry_d.get()))
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
            else:                       
                def func(tau,N,T,tau_T,tau_D):
                #add a case if any of the parameter is fixed 
                    Rsqd = 25
                    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
                best_vals, covariance = curve_fit(func, x, y, p0=initial_params[:4])
                errN,errT,errtau_T, errtau_D = np.sqrt(np.diag(covariance)) 
                error = [errN, errT,errtau_T,errtau_D]
                print('best val, error', best_vals,error) 
                self.best_val_list.append(best_vals) 
                self.error.append(error)
                return best_vals
            
        elif model == "Double Component Fitting":
             if fix_tau_T and fix_tau_D1:
                def func(tau,N,T,f,tau_D2):          
                        Rsqd = 25
                        tau_T = float(self.entry_c.get())
                        tau_D1 = float(self.entry_f.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[3],initial_params[5]])
                best_vals = np.insert(best_vals, [2,4],[float(self.entry_c.get()),float(self.entry_f.get())])
                errtau_T = 0
                errtau_D1 = 0
                errN,errT,errf,errtau_D2 = np.sqrt(np.diag(covariance))
                error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                print('best val', best_vals) 
                print('error',error ) 
                self.best_val_list.append(best_vals) 
                self.error.append(error)
                return best_vals
            
             if fix_N or fix_T or fix_tau_T or fix_f or fix_tau_D1 or fix_tau_D2 :
                print('hold param:')
                if fix_N:
                    def func(tau,T,tau_T,f,tau_D1,tau_D2):          
                        Rsqd = 25
                        N = float(self.entry_a.get())
                        #del initial_params[0]
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0= initial_params[1:6])
                    best_vals = np.insert(best_vals, 0,float(self.entry_a.get()))
                    errN = 0
                    errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    print('best val, error', best_vals, error) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_T:
                    def func(tau,N,tau_T,f,tau_D1,tau_D2):          
                        Rsqd = 25
                        T = float(self.entry_b.get())
                        #T = initial_params[1]
                        #del initial_params[1]
                        #print('T', T)
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[2],initial_params[3],initial_params[4],initial_params[5]])
                    best_vals = np.insert(best_vals, 1,float(self.entry_b.get()))
                    errT = 0
                    errN,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    #best_vals.insert[1,float(self.entry_b.get())]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_tau_T:
                    def func(tau,N,T,f,tau_D1,tau_D2):          
                        Rsqd = 25
                        tau_T = float(self.entry_c.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1], initial_params[3], initial_params[4],initial_params[5]])
                    best_vals = np.insert(best_vals, 2,float(self.entry_c.get()))
                    errtau_T = 0
                    errN,errT,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals)  
                    self.error.append(error)
                    return best_vals
                if fix_f:
                    def func(tau,N,T,tau_T,tau_D1,tau_D2):          
                        Rsqd = 25
                        f = float(self.entry_e.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[2],initial_params[4],initial_params[5]])
                    best_vals = np.insert(best_vals, 3,float(self.entry_e.get()))
                    errf = 0
                    errN,errT,errtau_T,errtau_D1,errtau_D2 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_tau_D1:
                    def func(tau,N,T,tau_T,f,tau_D2):          
                        Rsqd = 25
                        tau_D1 = float(self.entry_f.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[2],initial_params[3],initial_params[5]])
                    best_vals = np.insert(best_vals, 4,float(self.entry_f.get()))
                    errtau_D1 = 0
                    errN,errT,errtau_T,errf,errtau_D2 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                if fix_tau_D2:
                    def func(tau,N,T,tau_T,f,tau_D1):          
                        Rsqd = 25
                        tau_D2 = float(self.entry_g.get())
                        return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                    best_vals, covariance = curve_fit(func, x, y, p0=[initial_params[0],initial_params[1],initial_params[2],initial_params[3],initial_params[4]])
                    best_vals = np.insert(best_vals, 5,float(self.entry_g.get()))
                    errtau_D2 = 0
                    errN,errT,errtau_T,errf,errtau_D1 = np.sqrt(np.diag(covariance))
                    error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                    print('best val', best_vals) 
                    self.best_val_list.append(best_vals) 
                    self.error.append(error)
                    return best_vals
                
             else:                       
                def func(tau,N,T,tau_T,f,tau_D1,tau_D2):
                    Rsqd = 25
                    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))
                best_vals, covariance = curve_fit(func, x, y, p0=initial_params[:6])
                errN,errT,errtau_T,errf, errtau_D1,errtau_D2 = np.sqrt(np.diag(covariance))
                error =  [errN, errT,errtau_T,errf,errtau_D1,errtau_D2]
                print('best val, err', best_vals,error) 
                self.best_val_list.append(best_vals) 
                self.error.append(error)
                return best_vals
        
    def plot_graph(self):
        for index, self.FCSdata in enumerate(self.data_list):   
            x_data = self.FCSdata['time'].astype(float)
            y_data = self.FCSdata['corr1'].astype(float)
    #ToDo: remove file location 'D/' from label in the plot
            s = self.start_end_point[index][0]
            #print("S:", s)
            e = self.start_end_point[index][1]
            print("index:", index)
            plt.semilogx(x_data[s:e], y_data[s:e], marker='o',label = self.file_paths[index]) 
            plt.legend(loc="upper right")
            plt.show()
        plt.xlabel('T (s)', fontsize=16)
        plt.ylabel('g(T)', fontsize=16)    
        self.clear_loaded_data()     
        
        
    def update_graph(self):
        for index, self.FCSdata in enumerate(self.data_list):   
            x_data = self.FCSdata['time'].astype(float)
            y_data = self.FCSdata['corr1'].astype(float)
    #ToDo: remove file location 'D/' from label in the plot
            s = self.start_end_point[index][0]
            #print("S:", s)
            e = self.start_end_point[index][1]
            print("index:", index)
            plt.semilogx(x_data[s:e], y_data[s:e], marker='o',label = self.file_paths[index]) 
            plt.legend(loc="upper right")
            plt.show()
            
            N = self.entry_a.get()
            T = self.entry_b.get()
            tau_T = self.entry_c.get()
            tau_D = self.entry_d.get()
            f = self.entry_e.get()
            tau_D1 = self.entry_f.get()
            tau_D2 = self.entry_g.get()
#            N, T, tau_T, tau_D, f, tau_D1, tau_D2 = self.fix_param()
#            #print('param', N, T, tau_T, tau_D, f, tau_D1, tau_D2)
            #print(f"Entry 'N' value: {self.var_a.get()}")
            
            if self.model_var.get() == "Single Component Fitting":
                initial_params = [float(N),
                              float(T),
                              float(tau_T) ,
                              float(tau_D),]
                fitted_params = self.fit_curve(x_data[s:e], y_data[s:e], *initial_params, model=self.model_var.get())
                #print(f"Entry 'a' value: {self.entry_a.get()}")  
                print('initial_params',initial_params[:4])
            elif self.model_var.get() == "Double Component Fitting":
                initial_params = [float(N) ,
                              float(T) ,
                              float(tau_T) ,
                              float(f) ,
                              float(tau_D1) ,
                              float(tau_D2),]
                fitted_params = self.fit_curve(x_data[s:e], y_data[s:e], *initial_params, model=self.model_var.get())
            else:
                raise ValueError("Invalid model")
            print('fitted_params:', *fitted_params)    
            plt.plot(x_data[s:e], self.curve_model(x_data[s:e], *fitted_params), color='black', markersize=8, linestyle='-')
        plt.xlabel('T (s)', fontsize=16)
        plt.ylabel('g(T)', fontsize=16)
        self.size = fitted_params.shape[0]
        self.ind = index
        #self.mat = np.array(self.best_val_list, dtype=float).reshape(index+1, self.size)
        #print('size, mat', self.size, self.mat )    
            
    def curve_model(self, tau, *params):
        Rsqd = 25
        if self.model_var.get() == "Single Component Fitting":
            N,T,tau_T,tau_D = params
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)
        elif self.model_var.get() == "Double Component Fitting":
            N,T,tau_T,f,tau_D1,tau_D2 = params
        # Define the curve model based on the parameters
            return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5)) 
    
    def clear_loaded_data(self):
        self.data_list = []
        self.countrate_list = []
        self.file = []
        self.start_end_point = []
        self.best_val_list = [] 
        self.error = []
        self.data = None
        self.size = None
        self.mat = None
        self.errmat = None
        self.ind = None
             
         
    def load_data(self):       
        self.file_paths = filedialog.askopenfilenames(
                title="Select Data Files",
                filetypes=[("All files", "*.*")]
                )
        self.file_paths = list(self.file_paths)
       # self.file_paths.append(self.file_paths)
        for file_path in self.file_paths:
            if file_path.endswith(".ASC"):
        # Execute the following command for .asc files
                #print(f"Executing command for {file_path}")  
                #df = pd.read_csv(file_path, encoding='unicode_escape')
                df = pd.read_fwf(file_path, skiprows=30,encoding='unicode_escape')
                sz = len(df)
                df.columns = ['a','b','c','d','e']
                df = df.drop(['d','e'], axis = 1)
                idx1 = df.index[df['a'].str.contains('Corr') == True]
                idx2 = df.index[df['a'].str.contains('Count') == True]
                Correlation =df.iloc[ 0: idx2[0]-1].reset_index(drop=True)
                countrate = df.iloc[idx2[0]+1 : sz-2].reset_index(drop=True)
                FCSdata= pd.concat([Correlation,countrate], axis = 1)
                FCSdata.columns = ['time','corr1','corr2','countrate','intensity1','intensity2']              
                FCSdata['time'] = FCSdata['time'].astype(float)*0.001
                FCSdata = FCSdata.dropna()
                self.FCSdata = FCSdata
                self.data_list.append(self.FCSdata)
               # print("Data List:", self.data_list)
                self.file.append(file_path)
                #print("file_path:", self.file)
                avg_intensity = FCSdata['intensity1'].mean() + FCSdata['intensity2'].mean()  #KHz
                self.countrate_list.append(avg_intensity)
                #print("avg_intensity:", self.countrate_list)
                #a =[35,450]  
            #b = 450
                self.start_end_point.append([0,500]) # set start and end point of the dataset to be analysed;
                #print("start_end_point:", self.start_end_point)
            else:
                print(f"Executing command for {file_path}")
                df = pd.read_fwf(file_path, skiprows=16,encoding='unicode_escape')
                sz = len(df)
                df.columns = ['a','b','c']
                idx1 = df.index[df['a'].str.contains('Corr') == True]
                FCSdata = df.iloc[0:idx1[0]-1]
                FCSdata.columns = ['time','corr1','corr2']
                FCSdata = FCSdata.replace(0, np.nan)
                FCSdata =  FCSdata.dropna()
                FCSdata['corr1'] = FCSdata['corr1'].astype(float)-1
                self.data_list.append(FCSdata)
                self.file.append(file_path)
                dk = pd.read_fwf(file_path)
                dk.columns = ['a','b']
                idx_1 = dk.index[dk['a'].str.contains('Trace') == True]
                skip_row = idx_1[0]+5
                dk = pd.read_fwf(file_path, skiprows=skip_row)
                dk.columns = ['a']
                idx_2 = dk.index[dk['a'].str.contains('Histogram') == True]
                dk = dk.iloc[0:idx_2[0]-2]
                dk['a'] = dk['a'].str.replace(r'\s+', ' ', regex=True)
                dk['a'] = dk['a'].str.split(' ')
                dk[['col1', 'detector1', 'detector2']] = pd.DataFrame(dk['a'].tolist(), dtype=float)
                avg_intensity = dk['detector1'].mean()/1000 + dk['detector2'].mean()/1000  #KHz
                self.countrate_list.append(avg_intensity)
                #a = [85,450]
                self.start_end_point.append([65,450])  # set start and end point of the dataset to be analysed;
            

if __name__ == "__main__":
    root = tk.Tk()
    app = FCSFitting_app(root)
    root.mainloop()
        
    
    