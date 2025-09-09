import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
pi = np.pi
kb = 8.617333262145e-5  # Boltzmann constant in eV/K
T = 600 + 273.15  # Temperature in Kelvin

# Helper functions

def D_Coeff(D_0, E_m, T):
    """Calculate the diffusion coefficient."""
    return D_0 * np.exp(-E_m / (kb * T))

def log_factor(r_0, r_L, eps = 1e-10):
    """Compute the logarithmic factor for dislocation loops."""
    logt = np.log(8*r_L/r_0)
    r = r_0 * logt
    return 1/r if r > eps else eps

def compute_j_L_v(C_Ce_v, C_O_v, D_Ce_v, D_O_v, r_0, r_L):
    """Compute j_L^v based on the given concentrations and diffusion coefficients."""
    log_fact = log_factor(r_0, r_L)
    nominator = D_Ce_v * C_Ce_v * D_O_v * C_O_v
    denominator = 2 * D_Ce_v * C_Ce_v + D_O_v * C_O_v
    return log_fact * nominator / denominator

def compute_j_L_i(C_Ce_i, C_O_i, D_Ce_i, D_O_i, r_0, r_L):
    """Compute j_L^i based on the given concentrations and diffusion coefficients."""
    log_fact = log_factor(r_0, r_L)
    nominator = D_Ce_i * C_Ce_i * D_O_i * C_O_i
    denominator = 2 * D_Ce_i * C_Ce_i + D_O_i * C_O_i
    return log_fact * nominator / denominator

def compute_j_ii(C_Ce_i, C_O_i, D_Ce_i, D_O_i):
    """Compute j_ii based on the given concentrations and diffusion coefficients."""
    nominator = D_Ce_i * (C_Ce_i**2) * D_O_i * (C_O_i**2)
    denominator = 2 * D_Ce_i * (C_Ce_i**2) + D_O_i * (C_O_i**2)
    return nominator / denominator


# Main ODE system

def ODE_system(t, y, params):
    """Define the ODE system."""
    C_Ce_v, C_O_v, C_Ce_i, C_O_i, N_L, R_L = y

    # Unpack parameters
    G_VCe = params['G_Ce_v']
    G_VO = params['G_O_v']
    G_CeI = params['G_Ce_i']
    G_OI = params['G_O_i']
    a = params['a']
    Omega_0 = (a**3) / 12
    D_Ce_i = params['D_Ce_i']
    D_O_i = params['D_O_i']
    D_Ce_v = params['D_Ce_v']
    D_O_v = params['D_O_v']
    r0 = params['r0']
    b = a / np.sqrt(3)


    # Compute fluxes using separate functions
    j_L_v = compute_j_L_v(C_Ce_v, C_O_v, D_Ce_v, D_O_v, r0, R_L)
    j_L_i = compute_j_L_i(C_Ce_i, C_O_i, D_Ce_i, D_O_i, r0, R_L)
    j_ii = compute_j_ii(C_Ce_i, C_O_i, D_Ce_i, D_O_i)

    # Computing K_is
    k_ce = Omega_0 / a**2 * (48 * D_Ce_i + 48 * D_Ce_i)
    k_o = Omega_0 / a**2 * (36 * D_O_i + 24 * D_O_i)

    st = 2 * pi *R_L * N_L

    # ODEs
    dc = np.zeros(6)
    dc[0] = G_VCe - k_ce * C_Ce_v * C_Ce_i - j_L_v * pi * r0 * st
    dc[1] = G_VO - k_o * C_O_v * C_O_i - 2 * j_L_v * pi * r0 * st
    dc[2] = G_CeI - k_ce * C_Ce_i * C_Ce_v - j_L_i * pi * r0 * st - 84 * Omega_0 / a**2 * j_ii
    dc[3] = G_OI - k_o * C_O_i * C_O_v - 2 * j_L_i * pi * r0 * st - 2 * 84 * Omega_0 / a**2 * j_ii
    dc[4] = 84 * Omega_0 / a**2 * j_ii
    dc[5] = 3 * Omega_0 * 2 * pi * r0 / b * (j_L_i - j_L_v) - R_L / (2 * N_L) * 84 * Omega_0 / a**2 * j_ii

    return dc

# Plotting and data 

def plot_results(results):

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=False)

    axs[0, 0].loglog(results['time'], results['C_Ce_v']*1e7, label='C_Ce_v')
    axs[0, 0].loglog(results['time'], results['C_Ce_i']*1e7, label='C_Ce_i')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Concentration [m^-3]')
    axs[0, 0].legend()
    axs[0, 0].set_title('Vacancy Concentrations')

    axs[0, 1].loglog(results['time'], results['C_O_v']*1e7, label='C_O_v')
    axs[0, 1].loglog(results['time'], results['C_O_i']*1e7, label='C_O_i')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Concentration [m^-3]')
    axs[0, 1].legend()
    axs[0, 1].set_title('Interstitial Concentrations')

    axs[1, 0].plot(results['time'], results['N_L']*1e-6, label='N_L')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Loop Density [m^-3]')
    axs[1, 0].legend()
    axs[1, 0].set_title('Dislocation Loop Density')

    axs[1, 1].plot(results['time'], results['R_L']*1e7, label='R_L')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Loop Radius [nm]')
    axs[1, 1].legend()
    axs[1, 1].set_title('Dislocation Loop Radius')

    plt.tight_layout()
    plt.show()

# Example usage with scipy solver
if __name__ == "__main__":

    # Lattice and material parameters 
    a = 0.541e-7  # Lattice parameter in cm
    Omega_0 = (a**3) / 12  # Atomic volume in cm^3

    # Pre factors for diffusion coefficients (in cm^2/s)
    D_0_Ce_v = 0.65
    D_0_Ce_i = 0.01
    D_0_O_v = 0.02
    D_0_O_i = 0.01

    # Migration energies (in eV)
    E_m_Ce_v = 2.66
    E_m_Ce_i = 2.66
    E_m_O_v = 0.63
    E_m_O_i = 1.4

    # Generation rates (in dpa/s)
    # LF
    G_Ce_l = 0.87e-6
    # LH
    #G_Ce_l = 2.6e-6


    #Generation rates (in dpa/(s*cm^3))
    G_Ce = G_Ce_l / Omega_0
    G_O = 2 * G_Ce
    
    # Parameters
    params = {
        'G_Ce_v': G_Ce,
        'G_O_v': G_O,
        'G_Ce_i': G_Ce,
        'G_O_i': G_O,
        'a': a, 
        'D_Ce_i': D_Coeff(D_0_Ce_i, E_m_Ce_i, T),
        'D_Ce_v': D_Coeff(D_0_Ce_v, E_m_Ce_v, T),
        'D_O_i': D_Coeff(D_0_O_i, E_m_O_i, T),
        'D_O_v': D_Coeff(D_0_O_v, E_m_O_v, T),
        'r0': 3e-8,
    }
    print(f"D_Ce_i: {params['D_Ce_i']:.3e} cm^2/s")
    print(f"D_Ce_v: {params['D_Ce_v']:.3e} cm^2/s")
    print(f"D_O_i: {params['D_O_i']:.3e} cm^2/s")
    print(f"D_O_v: {params['D_O_v']:.3e} cm^2/s")
    # Initial conditions
    t0 = 1e-6
    y0 = [G_Ce*t0, G_O*t0, G_Ce*t0, G_O*t0, G_O*t0*1e-6, 2*Omega_0 / a**2  ]  

    # Time span
    t_span = (0, 1.2)  # Start and end time
    t_eval = np.linspace(0, 1.2, 1000)  # Time points for evaluation

    # Solve ODEs
    solution = solve_ivp(ODE_system, method='Radau', t_span=t_span, y0=y0, args=(params,), t_eval=t_eval)

    # Save results to a DataFrame
    results = pd.DataFrame(solution.y.T, columns=['C_Ce_v', 'C_O_v', 'C_Ce_i', 'C_O_i', 'N_L', 'R_L'])
    results['time'] = solution.t

    # Plot results
    plot_results(results)

