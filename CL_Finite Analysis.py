# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 01:20:56 2024

@author: rkuma
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.style as style


# Given values
T = 298  # Temperature in Kelvin
P = 101325  # Pressure in Pascal
R = 287  # Specific gas constant for air in J/(kg·K)

# Calculate density
rho = P / (R * T)
print(f"Density of air: {rho:.2f} kg/m³")

# Wing and flow characteristics
Uinf = 20  # Flow velocity (m/s)
b = 1.11  # Half-span of the wing (m)
c = 0.4  # Chord length (m)
bfull = 2 * b  # Full span of the wing
S = b * c  # Wing area (m²)
AR = bfull**2 / S  # Aspect ratio

# Calculate dynamic pressure
q_infinity = 0.5 * rho * Uinf**2
print(f"Dynamic pressure (q_inf): {q_infinity:.2f} Pa")

# Reference area (using half-span only)
A_ref = b * c  # Reference area in m²
print(f"Reference area (A_ref): {A_ref:.2f} m²")

# Read input data
input_csv_file = r"C:\Users\rkuma\Downloads\output_finite_wing.csv"  # Replace with your input file path
df = pd.read_csv(input_csv_file)

# Extract AoA, S, and D from the DataFrame
AoA = df['AoA (radians)'].values
Side = df['S'].values
D = -1*df['D'].values

# Compute true lift (L) and drag (D)
L = Side * np.cos(AoA) - D * np.sin(AoA)
D_true = D * np.cos(AoA) + Side * np.sin(AoA)

# Compute C_L and C_D
C_L = L / (q_infinity * S)
C_D = D_true / (q_infinity * S)

# Add AoA in degrees for clarity
AoA_degrees = np.degrees(AoA)

# Create a new DataFrame for output
output_data = {
    'AoA (degrees)': AoA_degrees,
    'C_L': C_L,
    'C_D': C_D
}
output_df = pd.DataFrame(output_data)

# Write the output DataFrame to a new CSV file
output_csv_file = 'output_CL_CD.csv'  # Replace with your desired file path
output_df.to_csv(output_csv_file, index=False)
print(f"Computed C_L and C_D have been written to {output_csv_file}")

# Define the linear function for fitting
def test_func(AoA, a, b):
    return a * AoA + b

# Filter the data for AoA <= 10 degrees
stall_limit = 10
valid_indices = AoA_degrees <= stall_limit
AoA_fit_degrees = AoA_degrees[valid_indices]
C_L_fit = C_L[valid_indices]

# Convert AoA_fit to radians for fitting
AoA_fit_radians = np.radians(AoA_fit_degrees)

# Perform curve fitting using AoA in radians
params_radians, params_covariance_radians = curve_fit(test_func, AoA_fit_radians, C_L_fit)

# Extract slope and intercept from the fit
slope_radians, intercept_radians = params_radians
print(f"Slope (radians): {slope_radians:.3f}")
print(f"Intercept: {intercept_radians:.3f}")


# Load the XFLR5 data
file_path = r"C:\Users\rkuma\Downloads\finitewing_xflr5.csv"  # Replace with your input file path
data = pd.read_csv(file_path)
# Extract data for plotting
AoA_xflr5 = data['alpha']
C_L_xflr5 = data['CL']

# Convert XFLR5 AoA to radians for fitting
AoA_xflr5_radians = np.radians(AoA_xflr5)

# Perform curve fitting using AoA in radians
params_xflr5, params_covariance_xflr5 = curve_fit(test_func, AoA_xflr5_radians, C_L_xflr5)

# Extract slope and intercept from the fit
slope_xflr5_radians, intercept_xflr5 = params_xflr5
print(f"Slope (radians): {slope_xflr5_radians:.3f}")
print(f"Intercept: {intercept_xflr5:.3f}")

# Convert slope back to per degree for interpretation
slope_xflr5_per_degree = slope_xflr5_radians * (np.pi / 180)
print(f"Slope (per degree): {slope_xflr5_per_degree:.3f}")


# Add the infinite wing lift-curve slope (2*pi in radians)
slope_infinite = 2 * np.pi  # Lift-curve slope for an infinite wing
# Generate a line for the infinite wing slope
AoA_range_degrees = np.linspace(min(AoA_degrees), 15, 100)
AoA_range_radians = np.radians(AoA_range_degrees)
C_L_infinite = slope_infinite * AoA_range_radians

# Using degrees for AoA in the plot


# clean style for better visualization
style.use('seaborn-colorblind')

plt.figure(figsize=(12, 8))

# Plot the experimental data
plt.scatter(
    AoA_degrees, C_L,
    color='blue', edgecolors='black', linewidths=1.5,
    s=100, marker='o', label='$C_L$ Data'
)

# Plot the best-fit line (experimental data)
plt.plot(
    AoA_fit_degrees, test_func(AoA_fit_radians, slope_radians, intercept_radians),
    linestyle='-', color='darkorange', linewidth=4,
    label=f'Best Fit (slope={slope_radians:.3f} rad$^{-1}$)'
)



# Plot the XFLR5 data points
plt.scatter(
    AoA_xflr5, C_L_xflr5,
    color='green', edgecolors='black', linewidths=1.5,
    s=120, marker='*', label='XFLR5 Data'
)

# Plot the best-fit line for XFLR5 data
plt.plot(
    AoA_xflr5, test_func(AoA_xflr5_radians, slope_xflr5_radians, intercept_xflr5),
    linestyle='-.', color='magenta', linewidth=3,
    label=f'XFLR5 Best Fit (slope={slope_xflr5_radians:.3f} rad$^{-1}$)'
)

# Customize the plot
plt.title('Lift Coefficient $C_L$ vs Angle of Attack $\\alpha$', fontsize=20, fontweight='bold')
plt.xlabel('Angle of Attack $\\alpha$ (degrees)', fontsize=16, fontweight='bold')
plt.ylabel('Lift Coefficient $C_L$', fontsize=16, fontweight='bold')

# Add major and minor gridlines
plt.grid(which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
plt.minorticks_on()

# Adjust the legend for better clarity
plt.legend(fontsize=14, loc='best', frameon=True, shadow=True)

# Enhance tick labels
plt.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tick_params(axis='both', which='minor', width=1, length=4)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()



# Filter data to exclude AoA > 15 degrees
valid_indices = AoA_degrees <= 15
C_D_filtered = C_D[valid_indices]
C_L_filtered = C_L[valid_indices]


C_D_xflr5 = data['CD']
C_L_xflr5 = data['CL']





# Define quadratic fit function
def drag_polar(C_L, C_D0, K):
    return C_D0 + K * C_L**2

# Perform curve fitting for experimental data
params_exp, _ = curve_fit(drag_polar, C_L_filtered, C_D_filtered)
C_D0_exp, K_exp = params_exp
e_exp = 1 / (np.pi * AR * K_exp)  # Oswald efficiency factor for experimental data

# Perform curve fitting for XFLR5 data
params_xflr5, _ = curve_fit(drag_polar, C_L_xflr5, C_D_xflr5)
C_D0_xflr5, K_xflr5 = params_xflr5
e_xflr5 = 1 / (np.pi * AR * K_xflr5)  # Oswald efficiency factor for XFLR5 data

# Generate a range of C_L values for plotting the fitted curves
C_L_range = np.linspace(min(C_L), max(C_L), 100)

# Compute fitted C_D values for both experimental and XFLR5 data
C_D_exp_fit = drag_polar(C_L_range, C_D0_exp, K_exp)
C_D_xflr5_fit = drag_polar(C_L_range, C_D0_xflr5, K_xflr5)


# Compute glide ratio (C_L / C_D)
glide_ratio = C_L_filtered / C_D_filtered

# Find the index of the maximum glide ratio
max_glide_index = np.argmax(glide_ratio)

# Extract the maximum glide ratio and corresponding angle of attack
max_glide_ratio = glide_ratio[max_glide_index]
max_glide_AoA = AoA_degrees[max_glide_index]

# Extract corresponding C_D and C_L_squared values
max_glide_C_D = C_D_filtered[max_glide_index]
max_glide_C_L = C_L_filtered[max_glide_index]

# Print the results
print(f"Maximum Glide Ratio: {max_glide_ratio:.4f}")
print(f"Angle of Attack for Maximum Glide Ratio: {max_glide_AoA:.2f} degrees")


# Compute glide ratio (C_L / C_D) for XFLR5 data
glide_ratio_xflr5 = C_L_xflr5 / C_D_xflr5

# Find the index of the maximum glide ratio for XFLR5 data
max_glide_index_xflr5 = np.argmax(glide_ratio_xflr5)

# Extract the maximum glide ratio and corresponding parameters
max_glide_ratio_xflr5 = glide_ratio_xflr5[max_glide_index_xflr5]
max_glide_AoA_xflr5 = AoA_xflr5[max_glide_index_xflr5]
max_glide_C_D_xflr5 = C_D_xflr5[max_glide_index_xflr5]
max_glide_C_L_xflr5 = C_L_xflr5[max_glide_index_xflr5]

# Print results
print(f"Maximum Glide Ratio (XFLR5): {max_glide_ratio_xflr5:.4f}")
print(f"Angle of Attack for Maximum Glide Ratio (XFLR5): {max_glide_AoA_xflr5:.2f} degrees")
print(f"Corresponding C_D (XFLR5): {max_glide_C_D_xflr5:.4f}")
print(f"Corresponding C_L (XFLR5): {max_glide_C_L_xflr5:.4f}")

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot of experimental data
plt.scatter(C_D_filtered, C_L_filtered, color='blue', edgecolors='black', s=100, label='Experimental Data')

# Plot fitted drag polar for experimental data
plt.plot(C_D_exp_fit, C_L_range, color='red', linewidth=2,
         label=f'Experimental Fit ($C_D = {C_D0_exp:.4f} + {K_exp:.4f}C_L^2$, $e = {e_exp:.4f}$)')

# Scatter plot of XFLR5 data
plt.scatter(C_D_xflr5, C_L_xflr5, color='green', edgecolors='black', s=120, marker='*', label='XFLR5 Data')

# Plot fitted drag polar for XFLR5 data
plt.plot(C_D_xflr5_fit, C_L_range, color='magenta', linewidth=2, linestyle='--',
         label=f'XFLR5 Fit ($C_D = {C_D0_xflr5:.4f} + {K_xflr5:.4f}C_L^2$, $e = {e_xflr5:.4f}$)')

# Customize the plot
plt.title('Drag Polar: $C_L$ vs $C_D$', fontsize=20, fontweight='bold')
plt.xlabel('Drag Coefficient $C_D$', fontsize=16, fontweight='bold')
plt.ylabel('Lift Coefficient $C_L$', fontsize=16, fontweight='bold')
plt.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=14, loc='best')
plt.tight_layout()

# Ensure the point is within the visible plot area
plt.xlim(min(C_D_filtered) - 0.05, max(C_D_filtered) + 0.05)
plt.ylim(min(C_L_filtered) - 0.05, max(C_L_filtered) + 0.05)

# Annotate the point of maximum glide ratio
plt.annotate(
    f'Max Glide Ratio\n$\\frac{{C_L}}{{C_D}} = {max_glide_ratio:.2f}$\n$\\alpha = {max_glide_AoA:.2f}^\circ$',
    xy=(max_glide_C_D, max_glide_C_L),  # Note swapped order: (x, y) = (C_D, C_L)
    xytext=(max_glide_C_D + 0.03, max_glide_C_L + 0.05),  # Adjusted offset for better visibility
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
    fontsize=12, color='black'
)

plt.annotate(
    f'Max Glide Ratio\n$\\frac{{C_L}}{{C_D}} = {max_glide_ratio_xflr5:.2f}$\n$\\alpha = {max_glide_AoA_xflr5:.2f}^\circ$',
    xy=(max_glide_C_D_xflr5, max_glide_C_L_xflr5),  # Note swapped order: (x, y) = (C_D, C_L)
    xytext=(max_glide_C_D_xflr5 + 0.1, max_glide_C_L_xflr5 + 0.2),  # Move annotation further away
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, color='black'),
    fontsize=12, color='black'
)

# Show the plot
plt.show()



# Function to calculate a based on the given formula for low aspect ratios.
def calculate_a(a0, AR):
    # Denominator of the formula
    denominator = np.sqrt(1 + (a0 / (np.pi * AR))**2) + (a0 / (np.pi * AR))
    # Calculate a
    a = a0 / denominator
    return a

# Example inputs
a0 = 2 * np.pi  # Infinite aspect ratio lift-curve slope (rad^-1, from theory or inviscid gradient)
a0_exp = 3.56

ARexp = b**2/S

# Calculate a theoretical
a = calculate_a(a0, AR)
print(f"Finite Wing Lift-Curve Slope theroretical (a): {a:.4f} rad^-1")

# Calculate a experimental
a_exp = calculate_a(a0_exp, AR)
print(f"Finite Wing Lift-Curve Slope theroretical (a_exp): {a_exp:.4f} rad^-1")

print("The aspect ratio is for xflr5", AR)
print("The aspect ratio is for experiment is", ARexp)

#  Function that calculates tau

def calculate_tau(a, a0, AR):
    """
    Calculate tau (τ) based on the given parameters:
    a  : Finite wing lift-curve slope (rad^-1)
    a0 : Infinite wing lift-curve slope (rad^-1)
    AR : Aspect ratio of the wing
    """
    # Solve for τ based on the given formula
    tau = ((((a0/a)  - 1)*np.pi*AR)/a0)-1
    return tau

#  inputs
a_experimental = 3.754        # Finite wing lift-curve slope (rad^-1) (from experimental data or fitting)
a0_exp =  3.56 # Infinite aspect ratio lift-curve slope (rad^-1, from experiment)
a_fromxflr5 = 4.34
a0_fromxfoil = 7.24

# Calculate tau
tau_experimental = calculate_tau(a_experimental, a0_exp, 1)
print(f"Efficiency factor τ from experiment: {tau_experimental:.4f}")

# Calculate tau
tau_xflr5 = calculate_tau(a_fromxflr5, a0_fromxfoil, AR)
print(f"Efficiency factor τ from XFLR5: {tau_xflr5:.4f}")


# Given Reynolds number
Reynolds = 103614
U = 24.67

# Empirical formula: C_F = 0.074 / (Re^0.2)
C_F_empirical = 0.074 / (Reynolds**0.2)

# Fitted curve: C_F = 0.1983 * (Re^-0.3033)
C_F_fitted = 0.1983 * (Reynolds**-0.3033)

C_F_empirical, C_F_fitted

# Calculating viscous drag for each CF value
Dv_empirical = C_F_empirical * 0.5 * rho * U**2 * S
Dv_fitted = C_F_fitted * 0.5 * rho * U**2 * S

print("the viscous drag using experimental relationship is",Dv_fitted )
print("the viscous drag using theoretical relationship is",Dv_empirical )

CD_total = 0.0316  # total drag coefficient

# Calculate form drag force
D_form = 0.5 * rho * U**2 * S * CD_total
print("The form drag is",D_form )

# Function to calculate a based on the given formula for low aspect ratios.
def calculate_a(a0, AR):
    # Denominator of the formula
    denominator = np.sqrt(1 + (a0 / (np.pi * AR))**2) + (a0 / (np.pi * AR))
    # Calculate a
    a = a0 / denominator
    return a

print("the low aspect ratio assumption a is",calculate_a(a0_exp, ARexp) )

print(D_form - Dv_fitted)

print(ARexp)


