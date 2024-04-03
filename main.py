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
    st.title("Kinetic simulator")
    st.subheader("You can choose between different set of conditions or try your own k values")
    st.text("K values are in mM and min as units.")
    st.text("k0: F➝W")
    st.text("k1: F + Ac ➝ O")
    st.text("k2: O ➝ An + W")
    st.text("k3: O ➝ Ac + W")
    st.text("k4: An ➝ Ac")
    st.text("We assume [O] ~ 0  so a = k3/k2")


    # Dictionary of parameter sets
    param_sets = {
        "Ac-FRGRGRGD-OH,  pH = 5.3, EDC": [4.5e-3, 0.0108, 0.4, .51],
        "Ac-FRGRGRGD-OH, pH = 5.3, DIC": [0.1764, 0.102, 1.34, 0.786],
        "Ac-D-OH, pH = 5.3, EDC": [4.5e-3, 0.0198, 1., 1.8],
        "Custom": None  # Placeholder for custom parameters
    }
    
    # Dropdown menu to select parameter set
    selected_set = st.selectbox("Select kinetic constant set", list(param_sets.keys()))

    # Check if "Custom" is selected
    if selected_set == "Custom":
        # Input fields for custom parameters
        st.write("## Custom Parameters")
        k0 = st.number_input("Enter k0", format="%.10f")
        k1 = st.number_input("Enter k1", format="%.10f")
        a = st.number_input("Enter a", format="%.10f")
        k4 = st.number_input("Enter k4", format="%.10f")
        params = [k0, k1, a, k4]
        st.write("Custom Parameters:", params)
    else:
        # Retrieve selected parameter set
        params = param_sets[selected_set]
        st.write("k0:", params[0])
        st.write("k1:", params[1])
        st.write("a:", params[2])
        st.write("k4:", params[3])
    


    k0 = params[0]
    k1 = params[1]
    a = params[2]
    k4 = params[3]

    # Input fields for the variables
    st.write("## Initial conditions")
    F = st.number_input("Enter EDC concentration [mM]", value=float(20), format="%.2f")
    Ac = st.number_input("Enter Precursor concentration [mM]", value=float(10), format="%.2f")

    t = st.number_input("Final simulation time [min]", value=200)

    tspan = np.linspace(0, t, 10000)

    initial_conditions = [F, Ac, 0, 0]
    params = [k0, k1, a, k4]

    return params, initial_conditions, tspan

def get_fitted_curve(initial_conditions, tspan, params):
    simulated = pd.DataFrame(odeint(kinetic_plotting, initial_conditions, tspan, args=(params,)), columns=['F', 'Ac', 'An', 'W'])
    simulated['min'] = tspan
    return simulated

def plot_simulation(simulated):

    colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#999999", "#F0E442", "#0072B2", "#D55E00"]
    palette = sns.color_palette(colors)

    sns.set_theme(context='notebook', style='ticks', 
                  font_scale=1.3, 
                  rc={"lines.linewidth": 1.6, 'axes.linewidth': 1.6, 
                      "xtick.major.width": 1.6, "ytick.major.width": 1.6}, 
                  palette=palette)



    fig, ax = plt.subplots(ncols=3, figsize=(8, 3), dpi = 200, sharex=True)

    ax[0].plot(simulated["min"], simulated["F"])
    ax[1].plot(simulated["min"], simulated["Ac"])
    ax[2].plot(simulated["min"], simulated["An"])

    ax[0].set_ylabel('EDC [mM]')
    ax[1].set_ylabel('Precursor [min]')
    ax[2].set_ylabel('Anhydride [min]')


    ax[0].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    ax[1].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    ax[2].set(xlabel = "Time [min]", xticks = np.linspace(0, simulated["min"].iloc[-1], 3))
    

    # Display the plot using Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    params, initial_conditions, tspan = streamlit_main()
    simulated_data = get_fitted_curve(initial_conditions, tspan, params)
    plot_simulation(simulated_data)

