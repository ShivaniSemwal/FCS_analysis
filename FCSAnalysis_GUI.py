# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:41:56 2023

@author: shivani
"""

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig
import pandas as pd
from tkinter import messagebox

# Create a Tkinter root window (it will not be displayed)
root = tk.Tk()
root.withdraw()

# Initialize an empty list to store selected file paths

        
def diff2(tau,N,T,tau_T,tau_D):
    Rsqd = 25
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*((1 + tau/tau_D)**(-1))*(1 + tau/(Rsqd*tau_D))**(-0.5)

def diff3(tau,N,T,tau_T,f,tau_D1,tau_D2):
    Rsqd = 25
    return (1 + (T/(1-T))*np.exp(-tau/tau_T))*(1/N)*(f*((1 + tau/tau_D1)**(-1))*(1 + tau/(Rsqd*tau_D1))**(-0.5)+(1-f)*((1 + tau/tau_D2)**(-1))*(1 + tau/(Rsqd*tau_D2))**(-0.5))

initial_params = {
    "diff2minus1": {'N': 1.50, 'T': 0.26, 'tau_T': 0.0088,'tau_D': 3.2},
    "diff3minus1": {'N': 4.0, 'T': 0.38, 'tau_T': 0.0088, 'f': 0.4,'tau_D1': 0.18, 'tau_D2': 0.3}
  #  "diff3minus1": {'N': N, 'T': T, 'tau_T': tau_T, 'f': 0.4,'tau_D1': 0.94*tau_D, 'tau_D2': 0.3}
}

def index_to_log_scale(index):
    # Convert an index to the corresponding value in log scale
    if 0 <= index < len(x_data):
        return x_data[index]
    return None

file_paths = filedialog.askopenfilenames(
        title="Select Data Files",
        filetypes=[("Text files", "*.asc")]
    )

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

        

       
        
# You can now use the selected file paths as needed.
def update_graph():
    
    selected_model = model_var.get()
    params = [float(param_entry.get()) for param_entry in param_entries[selected_model]]   
    global best_vals, N,T,tau_T,tau_D,f,tau_D1,tau_D2
    best_vals,N,T,tau_T,tau_D,f,tau_D1,tau_D2 = [],[],[],[],[],[],[],[]
    start = 1
    end = len(x_data)-1
    try: 
        start = float(start_entry.get())
        end = float(end_entry.get())
    except Exception as e:
        messagebox.showerror("Error", f"Curve fitting failed: {str(e)}")
    start_val = index_to_log_scale(start)
    end_val = index_to_log_scale(end)    
    
     #Filter data within the specified range
    x_filtered = x_data[(x_data >= start_val) & (x_data <= end_val)]
    #y_filtered = y_data[(x_data >= start) & (x_data <= end)]    
    
    if selected_model == "diff2minus1":
        best_vals, covar = curve_fit(diff2, x_data, y_data, params)
        N,T,tau_T,tau_D = best_vals
        errN,errT,errtau_T,errtau_D = np.sqrt(np.diag(covar))
        fitted_data = diff2(x_filtered, *best_vals)
        #fitted_data = diff2(x_data, *best_vals)
        textstr = '\n'.join((
                r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                #  r'$Rsqd=%.2f$' % (Rsqd, ),
                r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                r'$\tau_D=%.4f$ $\pm$ %.4f' % (tau_D, errtau_D )))

       # status_label.config('best_vals: {}'.format(best_vals))
    elif selected_model == "diff3minus1":   
        best_vals, covar = curve_fit(diff3 ,x_data, y_data, params)
        N,T,tau_T,f,tau_D1,tau_D2 = best_vals
        errN,errT,errtau_T,errf,errtau_D1,errtau_D2 = np.sqrt(np.diag(covar))
        fitted_data = diff3(x_filtered, *best_vals)
        #fitted_data = diff3(x_data, *best_vals)
        textstr = '\n'.join((
                r'$N=%.4f$ $\pm$ %.4f' % (N, errN),
                #  r'$Rsqd=%.2f$' % (Rsqd, ),
                r'$T=%.2f$ $\pm$ %.4f' % (T, errT ),
                r'$\tau_T=%.4f$ $\pm$ %.4f' % (tau_T, errtau_T ),
                #  r'$\tau_D=%.4f$ $\pm$ %.4f' % (tau_D, )))
                r'$f=%.2f$ $\pm$ %.4f' % (f, errf ),
                r'$\tau_D1=%.4f$ $\pm$ %.4f' % (tau_D1, errtau_D1 ),
                r'$\tau_D2=%.4f$ $\pm$ %.4f' % (tau_D2, errtau_D2 )))
            #status_label.config('best_vals: {}'.format(best_vals))
    else:
        return  # No valid model selected
    
    ax.clear()
    
    ax.plot(x_data, y_data, label='Data', color='blue')
    ax.plot(x_filtered, fitted_data, label=f'Fitted {selected_model}', color='red')
    #ax.plot(x_data[start:end], fitted_data, label=f'Fitted {selected_model}', color='red')
    
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
    ax.text(0.55, 0.70, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
    canvas.draw()
    
# Create a function to save the plot
def save_plot():
    # Create a file dialog to choose the save location and filename
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    
    if file_path:
        # Save the current figure to the specified file path
        fig.savefig(file_path, dpi=fig.dpi)
        print(f"Plot saved to {file_path}")
        
def print_values():    
    #label.config(text=f"best_val: {best_vals}")  
     
     label.config(text=f"N: {N},T: {T},tau_t: {tau_T}, tau_D: {tau_D}, f: {f},tau_D1: {tau_D1},tau_D2: {tau_D2} ")        


        
def load_and_plot_data():
    
    file_paths = filedialog.askopenfilenames(
        title="Select Data Files",
        filetypes=[("Text files", "*.asc")]
    )
    
    if not file_paths:
        return  # No files selected
    
    global x_data, y_data, sz_x, sz_y, x_filtered
    x_data, y_data, sz_x, sz_y ,x_filtered = [], [],[],[],[]
    
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
        sz_x = len(x_data)-1
        sz_y = len(y_data)-1
    
    update_graph()
root = tk.Tk()
root.title("Fit Data with Fitting Functions")

# Create a label and dropdown to select the fitting function
model_label = ttk.Label(root, text="Select Fitting Function:")
model_label.pack()    

models = ["diff2minus1", "diff3minus1"]  # Add more fitting functions if needed
model_var = tk.StringVar()
model_var.set(models[0])
model_dropdown = tk.OptionMenu(root, model_var, *models)
model_dropdown.pack()

def data_size():
#    sz_x = len(x_data)
#    sz_y = len(y_data)
    label.config(text=f"x_data:{sz_x}, y_data:{sz_y}") 
    
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
load_button = ttk.Button(root, text="plot_selected_files", command=load_and_plot_data)
load_button.pack()

frame = ttk.Frame(root)
frame.pack()
print_button = ttk.Button(frame, text="Print fitted data", command=print_values)
print_button.grid()

#label = ttk.Label(frame, text="")
#label.grid()

frame = ttk.Frame(root)
frame.pack()
print_button = ttk.Button(frame, text="data size", command=data_size)
print_button.grid()

label = ttk.Label(frame, text="")
label.grid()

# Create a Matplotlib figure and canvas
#fig, ax = plt.subplots(figsize=(8, 6))
#canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.get_tk_widget().pack()

# Create a matplotlib figure and canvas to display the graph
mpl_fig.Figure(figsize=(6, 4), dpi=100)
fig = plt.Figure()
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Create entry fields for defining the start and end points
start_label = ttk.Label(root, text="Start Point:")
start_label.pack()
start_entry = ttk.Entry(root)
start_entry.insert(0, "1")  # Default start value
start_entry.pack()

end_label = ttk.Label(root, text="End Point:")
end_label.pack()
end_entry = ttk.Entry(root)
end_entry.insert(0, len(x_data)) 
#end_entry.insert(0, "231.0")  # Default end value
end_entry.pack()

# Create a button to save the plot
save_button = ttk.Button(root, text="Save Plot", command=save_plot)
save_button.pack()

# Update the graph initially
update_graph()


# Start the tkinter main loop
root.quit()


