import streamlit as st
import pandas as pd


from scipy.integrate import odeint, solve_ivp, lsoda
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, Model, report_fit, conf_interval
from sklearn.metrics import mean_squared_error
import numdifftools
from PIL import Image
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scipy.optimize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tkr


def kinetic_plotting(z, t, params):
    F, Ac, An, W = z  
    k0, k1, a, k4 = params
    
    dAcdt = -k1*F*Ac + k4*An + ((k1*a*Ac*F)/(a+1))
    dFdt = -k1*F*Ac - k0*F
    dWdt = +k1*F*Ac + k0*F
    dAndt = +((k1*Ac*F)/(a+1)) - k4*An
    
    return [dFdt, dAcdt, dAndt, dWdt]

def streamlit_main():
    st.title("Variable Input")
    
    # Input fields for the variables
    st.write("## Kinetic constants")

    k0 = st.number_input("Enter k0", value=1.8e-4, format="%.10f")
    k1 = st.number_input("Enter k1", value=4.79e-3, format="%.10f")
    a = st.number_input("Enter a", value=0.3, format="%.10f")
    k4 = st.number_input("Enter k4", value=8.38e-1, format="%.10f")

    # Input fields for the variables
    st.write("## Initial conditions")
    F = st.number_input("Enter EDC concentration [mM]", value=float(20), format="%.2f")
    Ac = st.number_input("Enter Precursor concentration [mM]", value=float(10), format="%.2f")

    
    t = st.number_input("Final simulation time [min]", value=200)

    tspan = np.linspace(0, t, 10000)


    params = [k0, k1, a, k4]
    initial_conditions = [F, Ac, 0, 0]

    return params, initial_conditions, tspan

def get_fitted_curve(initial_conditions, tspan, params):
    simulated = pd.DataFrame(odeint(kinetic_plotting, initial_conditions, tspan, args=(params,)), columns=['F', 'Ac', 'An', 'W'])
    simulated['min'] = tspan
    return simulated

def plot_simulation(simulated):

    colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#999999", "#F0E442", "#0072B2", "#D55E00"]
    palette = sns.color_palette(colors)

    sns.set_theme(context='notebook', style='ticks', font='Arial', 
                  font_scale=1.3, 
                  rc={"lines.linewidth": 1.6, 'axes.linewidth': 1.6, 
                      "xtick.major.width": 1.6, "ytick.major.width": 1.6}, 
                  palette=palette)



    fig, ax = plt.subplots(ncols=3, figsize=(8, 2.5), dpi = 200, sharex=True)

    ax[0].plot(simulated["min"], simulated["F"])
    ax[1].plot(simulated["min"], simulated["Ac"])
    ax[2].plot(simulated["min"], simulated["An"])

    ax[0].set_ylabel('EDC [mM]')
    ax[1].set_ylabel('Precursor [min]')
    ax[2].set_ylabel('Anhydride [min]')


    ax[0].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    ax[1].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    ax[2].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    params, initial_conditions, tspan = streamlit_main()
    simulated_data = get_fitted_curve(initial_conditions, tspan, params)
    plot_simulation(simulated_data)

