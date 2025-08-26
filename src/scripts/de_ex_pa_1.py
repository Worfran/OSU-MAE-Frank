import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
from tqdm import tqdm

# Base constants 
pi = np.pi
a = 5e-10
S0 = a**3/4
Z = 84
alpha = Z*S0/a**2
kb = 8.62e-5
nu = 2e13
NL = 1e-7/S0
b = a/np.sqrt(3)
r0 = 4*a 
rL_0 = a

# Define the problem for SALib
problem = {
    'num_vars': 4,
    'names': ['T', 'Ei', 'Ev', 'G_0'],
    'bounds': [
        [873.15, 1273.15],    # Temperature range (K)
        [0.5, 1.5],           # Interstitial migration energy (eV)
        [0.5, 1.5],           # Vacancy migration energy (eV) 
        [1e-7, 1e-5]          # Generation rate range
    ]
}

def log_term(y, r0):
    """Calculate the logarithmic term."""
    return np.log(8 * y / r0)

def ode_func_parametric(T, Ei, Ev, G_0):
    """
    Modified ODE function that accepts parameters for sensitivity analysis.
    """
    def ode_func(t, y):
        # Calculate dependent parameters
        wi = nu * np.exp(-Ei/(kb*T))
        wv = nu * np.exp(-Ev/(kb*T))
        Di = (a**2) * wi
        Dv = (a**2) * wv
        f1 = alpha * (Di + Dv)
    
        dc = np.zeros(3)
    
        log_val = log_term(y[2], r0)
        Kis = Di * 2 * pi**2 * y[2] / log_val
        Kvs = Dv * 2 * pi**2 * y[2] / log_val
        
        # ODE equations
        dc[0] = G_0 - f1 * y[0] * y[1] - Kis * y[0] * NL
        dc[1] = G_0 - f1 * y[0] * y[1] - Kvs * y[1] * NL
        dc[2] = (S0 * pi / (b * log_val)) * (Di * y[0] - Dv * y[1])

        return dc

    return ode_func

def solve_ode_parametric(t_span, y0, T, Ei, Ev, G_0):
    """Solve ODE with given parameters."""
    ode_function = ode_func_parametric( T, Ei, Ev, G_0)
    return solve_ivp(ode_function, 
                    t_span, y0, method='Radau', 
                    t_eval=np.logspace(0, 5, 1000))


def generate_samples(problem, N=64):
    """Generate parameter samples."""
    param_values = saltelli.sample(problem, N)
    print(f"Generated {len(param_values)} parameter combinations")
    return param_values

def run_model(params):
    """Run model and return output."""
    T, Ei, Ev, G_0 = params
    
    try:
        t_span = (0, 1e5)
        y0 = [0, 0, rL_0]
        sol = solve_ode_parametric(t_span, y0, T, Ei, Ev, G_0)

        return (sol.y[0][-1], sol.y[1][-1], sol.y[2][-1] * 1e9)

    except:
        return np.nan

def run_analysis(problem, N=64):
    """
    Main function that runs the complete sensitivity analysis.
    Returns all data needed for plotting.
    """
    
    # Generate parameter samples
    param_values = generate_samples(problem, N)
    
    # Run model for all parameter combinations
    results = []
    
    for i, params in enumerate(tqdm(param_values, desc="Simulations")):
        result = run_model(params)
        results.append(result)
    
    # Convert to arrays and DataFrames
    results_array = np.array(results)
    


    df_results = pd.DataFrame(results_array, columns=['C_i', 'C_v', 'R_L'])

    
    df_params = pd.DataFrame(param_values, columns=problem['names'])
    
    # Calculate correlations
    correlation_matrix = []
    for output_col in df_results.columns:
        correlations = []
        for param_col in df_params.columns:
            
            valid_mask = ~(np.isnan(df_results[output_col]) | np.isnan(df_params[param_col]))
            if valid_mask.sum() > 10: 
                corr = np.corrcoef(df_params.loc[valid_mask, param_col], 
                                 df_results.loc[valid_mask, output_col])[0, 1]
            else:
                corr = np.nan
            correlations.append(corr)
        correlation_matrix.append(correlations)
    
    correlation_matrix = np.array(correlation_matrix)
    
    return df_params, df_results, correlation_matrix

def heatmap_plotting(correlation_matrix, param_names, output_names):
    """
    Create heatmap showing only correlation coefficients.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create correlation heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_title('Parameter-Output Correlations', fontsize=16)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticks(range(len(output_names)))
    ax.set_yticklabels(output_names)
    
    # Add correlation values on heatmap
    for i in range(len(output_names)):
        for j in range(len(param_names)):
            if not np.isnan(correlation_matrix[i, j]):
                text_color = 'white' if abs(correlation_matrix[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                       ha="center", va="center", color=text_color, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha="center", va="center", color='gray')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    
    plt.tight_layout()
    plt.savefig('de-ex-pa-1-heatmap.svg', format='svg', dpi=600)
    plt.show()
    

def plotting_func(df_params, df_results, correlation_matrix):
    """
    Create additional plots to visualize parameter-output relationships.
    """
    n_params = len(df_params.columns)
    n_outputs = len(df_results.columns)
    
    # Create scatter plots showing parameter vs output relationships
    fig, axes = plt.subplots(n_outputs, n_params, figsize=(4*n_params, 3*n_outputs))
    
    
    for i, output_col in enumerate(df_results.columns):
        for j, param_col in enumerate(df_params.columns):
            ax = axes[i, j]
            
            # Get data and remove NaN values
            x_data = df_params[param_col].values
            y_data = df_results[output_col].values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            
            if valid_mask.sum() > 0:
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]
                
                # Scatter plot
                ax.scatter(x_clean, y_clean, alpha=0.6, s=20)
                ax.set_xlabel(param_col)
                ax.set_ylabel(output_col)
                ax.grid(True, alpha=0.3)
                
                # Add correlation value and trend line if correlation is strong
                corr = correlation_matrix[i, j]
                if not np.isnan(corr):
        
                    # Add trend line if correlation is significant
                    if abs(corr) > 0.3:
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2, label=f'Correlation = {corr:.3f}')
                        ax.legend()
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_xlabel(param_col)
                ax.set_ylabel(output_col)
    
    plt.suptitle('Parameter-Output Relationships', fontsize=16)
    plt.tight_layout()
    plt.savefig('de-ex-pa-1-param-output.svg', format='svg', dpi=600)
    plt.show()
    

# Update your main execution:
if __name__ == "__main__":
    # Run the complete analysis
    df_params, df_results, correlation_matrix = run_analysis(problem, N=5)

    # Heatmap
    heatmap_plotting(correlation_matrix, problem['names'], df_results.columns.tolist())
    #Linear plots
    plotting_func(df_params, df_results, correlation_matrix)
