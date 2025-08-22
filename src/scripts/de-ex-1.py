#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Constants
pi = np.pi

a=5e-10
S0=a**3/4
Z=84
alpha=Z*S0/a**2

T=600
kb=8.62e-5
Ei=1.2
Ev=1.8
nu=2e13
wi=nu*np.exp(-Ei/(kb*T))
wv=nu*np.exp(-Ev/(kb*T))
Di=(a**2)*wi
Dv=(a**2)*wv

G_0 = 1e-6/S0

NL=1e-7/S0
b=a/np.sqrt(3)
r0=4*a 
rL_0=a

#ODE function
f1 = alpha * (Di + Dv)

def log_term(y):
    """Calculate the logarithmic term.
    Args:
        y (float): The value R_L for which to calculate the logarithm.
    Returns:
        float: The logarithmic term.
    """
    return np.log(8 * y / r0)

def ode_func(t, y):
    """
    Define the system of ODEs.
    Args:
        t (float): Current time.
        y (list): Current values of the system [C_i, C_v, R_L].
    Returns:
        np.ndarray: The derivatives of the system [dC_i/dt, dC_v/dt, dR_L/dt].
    """
    dc = np.zeros(3)
    
    log_val = log_term(y[2])
    Kis = Di * 2 * pi**2 * y[2] / log_val
    Kvs = Dv * 2 * pi**2 * y[2] / log_val
    
    # Fixed ODE equations
    dc[0] = G_0 - f1 * y[0] * y[1] - Kis * y[0] * NL
    dc[1] = G_0 - f1 * y[0] * y[1] - Kvs * y[1] * NL
    dc[2] = (S0 * pi / (b * log_val)) * (Di * y[0] - Dv * y[1])
    return dc

def solve_ode(t_span, y0):
    """Solve the ODE using scipy's solve_ivp.
    Args:
        t_span (tuple): The time span for the solution.
        y0 (list): Initial conditions.
    Returns:
        OdeResult: The result of the ODE solution.
    """


    start_time = max(t_span[0], 1e-3)  
    end_time = t_span[1]
    
    # Create logarithmic spacing from start_time to end_time
    t_eval = np.logspace(np.log10(start_time), np.log10(end_time), 1000)

    return solve_ivp(ode_func, t_span, y0, method='Radau', 
                    rtol=1e-8, atol=1e-11,
                    t_eval=t_eval)

#plotting function
def plot_results(sol):
    """Plot the results of the ODE solution.
    Args:
        sol (OdeResult): The result of the ODE solution.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Log-log plot of concentrations (matching MATLAB style)
    axs[0].loglog(sol.t, sol.y[0] * S0, color='blue', linewidth=2, label='Interstitial')
    axs[0].loglog(sol.t, sol.y[1] * S0, color='orange', linewidth=2, label='Vacancy')
    axs[0].set_ylabel('Concentration')
    axs[0].set_xlabel('Time')
    axs[0].set_title('Defect Concentrations vs Time')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Radius evolution
    axs[1].plot(sol.t, sol.y[2]*1e9, color='green', linewidth=2)
    axs[1].set_ylabel('Radius [nm]')
    axs[1].set_title('Loop Radius vs Time')
    axs[1].set_xlabel('Time')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    t_span = (0, 1e5)  # Time span for the ODE solution
    y0 = [0, 0, rL_0]  # Initial conditions
    sol = solve_ode(t_span, y0)  # Solve the ODE
    plot_results(sol)  # Plot the results