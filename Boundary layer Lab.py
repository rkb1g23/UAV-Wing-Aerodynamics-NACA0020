# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 00:05:27 2024

@author: rkuma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import cm

rho_air = 1.25
nu_air = 1.5e-5
mu_air = rho_air * nu_air
rho_water = 1000  # kg/m^3, density of water
g = 9.81  # m/s^2, gravitational acceleration

# Manometer readings in mm
total_manometer_1 = 188  # mm
static_manometer_1 = 174  # mm

# Convert manometer readings to pressure (Pa)
delta_P_total = total_manometer_1 / 1000 * rho_water * g  # Total pressure in Pa
delta_P_static = static_manometer_1 / 1000 * \
    rho_water * g  # Static pressure in Pa

# Calculate dynamic pressure
delta_P_dynamic = delta_P_total - delta_P_static

# Calculate velocity (v) using dynamic pressure
v = (2 * delta_P_dynamic / rho_air) ** 0.5  # m/s

# characteristic length (L)
L = 0.265  # m

# Calculate Reynolds number
Re = v * L / nu_air

# Display results
print(Re)


# Position vs Velocity Files for different Reynold numbers
Case1_Re_261886 = pd.read_csv(r"C:\Users\rkuma\Downloads\case1.csv")
Case2_Re_194333 = pd.read_csv(r"C:\Users\rkuma\Downloads\BLdata-0.25.csv")
Case3_Re_335666 = pd.read_csv(r"C:\Users\rkuma\Downloads\BLdata2-0.66.csv")
Case4_Re_540600 = pd.read_csv(r"C:\Users\rkuma\Downloads\Re_540600.csv")
Case5_Re_441666 = pd.read_csv(r"C:\Users\rkuma\Downloads\Re_441666.csv")

# Data repeats especifically used for CF and Re curve fit
Case6_Re_192566 = pd.read_csv(r"C:\Users\rkuma\Downloads\0.25 throttle me.csv")
Case7_Re_602786 = pd.read_csv(r"C:\Users\rkuma\Downloads\0.66 throttle me.csv")
Case8_Re_462866 = pd.read_excel(r"C:\Users\rkuma\Downloads\2 Thirds speed BL.xlsx")
Case9_Re_143100 = pd.read_excel(r"C:\Users\rkuma\Downloads\quarter wing speed.xlsx")
Case10_Re_509683 = pd.read_csv(r"C:\Users\rkuma\Downloads\BLlab2024 0.75.csv")
Case11_Re_538833 = pd.read_csv(r"C:\Users\rkuma\Downloads\full_throttle.csv")
Case12_Re_415166 = pd.read_csv(r"C:\Users\rkuma\Downloads\half_throttle.csv")
Case13_Re_215886 = pd.read_csv(r"C:\Users\rkuma\Downloads\0.3333 throttle.csv")
Case14_Re_503500 = pd.read_csv(r"C:\Users\rkuma\Downloads\BLLab_0.750.csv")
Case15_Re_226133 = pd.read_csv(r"C:\Users\rkuma\Downloads\0.3 throttle (2).csv")
Case16_Re_348033 = pd.read_csv(r"C:\Users\rkuma\Downloads\3quarters_throttle case (1).csv")

# Extract columns for Case1_Re_261886

aa, ab = Case1_Re_261886.columns  # column names
y_Re_261886 = Case1_Re_261886[aa]
u_Re_261886 = Case1_Re_261886[ab]
Uinf_Re_261886 = u_Re_261886.max()  # We get the freestream for Case1_Re_261886

# Extract columns from Case2_Re_194333
ba, bb = Case2_Re_194333.columns  # column names
y_Re_194333 = Case2_Re_194333[ba]
u_Re_194333 = Case2_Re_194333[bb]
Uinf_Re_194333 = u_Re_194333.max()  # Freestream velocity for Case2_Re_194333

# Extract columns from Case3_Re_335666
ca, cc = Case3_Re_335666  # column names
y_Re_335666 = Case3_Re_335666[ca]
u_Re_335666 = Case3_Re_335666[cc]
Uinf_Re_335666 = u_Re_335666.max()  # Freestream velocity for Case3_Re_335666

# Extract columns from Case4_Re_540600
ac, cd = Case4_Re_540600  # column names
y_Re_540600 = Case4_Re_540600[ac]
u_Re_540600 = Case4_Re_540600[cd]
Uinf_Re_540600 = u_Re_540600.max()  # Freestream velocity for Case4_Re_540600

# Extract columns from Case5_Re_441666
da, db = Case5_Re_441666  # column names
y_Re_441666 = Case5_Re_441666[da]
u_Re_441666 = Case5_Re_441666[db]
Uinf_Re_441666 = u_Re_441666.max()  # Freestream velocity for Case5_Re_441666

# Extract columns for Case6_Re_192566

ff, fb = Case6_Re_192566.columns  # column names
y_Re_192566 = Case6_Re_192566[ff]
u_Re_192566 = Case6_Re_192566[fb]
Uinf_Re_192566 = u_Re_192566.max()  # We get the freestream for Case6_Re_192566

# Extract columns from Case7_Re_602786
fd, df = Case7_Re_602786.columns  # column names
y_Re_602786 = Case7_Re_602786[fd]
u_Re_602786 = Case7_Re_602786[df]
Uinf_Re_602786 = u_Re_602786.max()  # Freestream velocity for Case7_Re_602786

# Extract columns from Case8_Re_462866
md, dm = Case8_Re_462866.columns  # column names
y_Re_462866 = Case8_Re_462866[md]
u_Re_462866 = Case8_Re_462866[dm]
Uinf_Re_462866 = u_Re_462866.max()  # Freestream velocity for Case8_Re_462866

# Extract columns from Case9_Re_143100
mc, cm = Case9_Re_143100.columns  # column names
y_Re_143100 = Case9_Re_143100[mc]
u_Re_143100 = Case9_Re_143100[cm]
Uinf_Re_143100 = u_Re_462866.max()  # Freestream velocity for Case9_Re_143100

# Extract columns from Case10_Re_509683
mf, fm = Case10_Re_509683.columns  # column names
y_Re_509683 = Case10_Re_509683[mf]
u_Re_509683 = Case10_Re_509683[fm]
Uinf_Re_509683 = u_Re_509683.max()  # Freestream velocity for Case10_Re_509683

# Extract columns from Case11_Re_538833
nm, mn = Case11_Re_538833.columns  # column names
y_Re_538833 = Case11_Re_538833[nm]
u_Re_538833 = Case11_Re_538833[mn]
Uinf_Re_538833 = u_Re_538833.max()  # Freestream velocity for Case11_Re_538833

# Extract columns from Case12_Re_415166
nf, fn = Case12_Re_415166.columns  # column names
y_Re_415166 = Case12_Re_415166[nf]
u_Re_415166 = Case12_Re_415166[fn]
Uinf_Re_415166 = u_Re_415166.max()  # Freestream velocity for Case12_Re_415166

# Extract columns for Case13_Re_215886

zy, yz = Case1_Re_261886.columns  # column names
y_Re_215886 = Case13_Re_215886[zy]
u_Re_215886 = Case13_Re_215886[yz]
Uinf_Re_215886 = u_Re_215886.max()  # We get the freestream for Case13_Re_215886

# Extract columns for Case14_Re_503500

zx, xz= Case14_Re_503500.columns  # column names
y_Re_503500 = Case14_Re_503500[zx]
u_Re_503500 = Case14_Re_503500[xz]
Uinf_Re_503500 = u_Re_503500.max()  # We get the freestream for Case14_Re_503500

# Extract columns for Case15_Re_226133

zd, dz= Case15_Re_226133.columns  # column names
y_Re_226133 = Case15_Re_226133[zd]
u_Re_226133 = Case15_Re_226133[dz]
Uinf_Re_226133 = u_Re_226133.max()  # We get the freestream for Case15_Re_226133

yx, xy= Case16_Re_348033.columns  # column names
y_Re_348033 = Case16_Re_348033[yx]
u_Re_348033 = Case16_Re_348033[xy]
Uinf_Re_348033 = u_Re_348033.max()  # We get the freestream for Case16_Re_348033



# Enhanced Plot for Velocity Profiles
plt.figure(figsize=(10, 8))

# Plot in order from low Re to high Re
plt.plot(u_Re_194333, y_Re_194333, 'x-', label='Re = 194333', markersize=6, linewidth=1.5)
plt.plot(u_Re_261886, y_Re_261886, 'o-', label='Re = 261886', markersize=6, linewidth=1.5)
plt.plot(u_Re_335666, y_Re_335666, '+-', label='Re = 335666', markersize=6, linewidth=1.5)
plt.plot(u_Re_441666, y_Re_441666, 'p-', label='Re = 441666', markersize=6, linewidth=1.5)
plt.plot(u_Re_540600, y_Re_540600, 'd-', label='Re = 540600', markersize=6, linewidth=1.5)



# Labels and formatting
plt.xlabel("Velocity (u) [m/s]", fontsize=14, labelpad=10)
plt.ylabel("Position (y) [mm]", fontsize=14, labelpad=10)
plt.title("Velocity Profiles for Different Reynolds Numbers", fontsize=16, pad=15)
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

# Find the Boundary layer thickness for CASE 1
delta_Re_261886 = np.interp(0.99, u_Re_261886/Uinf_Re_261886, y_Re_261886)
print("BL thickness Case1_Re_261886 (mm):", np.round(delta_Re_261886, 3))

# Calculate the gradient at the wall for wall-shear-stress
tau_w_Re_261886 = 1000 * mu_air * \
    (u_Re_261886[1] - u_Re_261886[0]) / (y_Re_261886[1] - y_Re_261886[0])
print(" wall shear stress Case1_Re_261886 (Pa):", tau_w_Re_261886)

# Find the Boundary layer thickness for CASE 2
delta_Re_194333 = np.interp(0.99, u_Re_194333/Uinf_Re_194333, y_Re_194333)
print("BL thickness Case2_Re_194333 (mm):", np.round(delta_Re_194333, 3))

# Calculate the gradient at the wall for wall-shear-stress
tau_w_Re_194333 = 1000 * mu_air * \
    (u_Re_194333[1] - u_Re_194333[0]) / (y_Re_194333[1] - y_Re_194333[0])
print(" wall shear stress Case2_Re_194333 (Pa):", tau_w_Re_194333)

# Find the Boundary layer thickness for CASE 3
delta_Re_335666 = np.interp(0.99, u_Re_335666/Uinf_Re_335666, y_Re_335666)
print("BL thickness Case3_Re_335666 (mm):", np.round(delta_Re_335666, 3))

# Calculate the gradient at the wall for wall-shear-stress
tau_w_Re_335666 = 1000 * mu_air * \
    (u_Re_335666[1] - u_Re_335666[0]) / (y_Re_335666[1] - y_Re_335666[0])
print(" wall shear stress Case3_Re_335666 (Pa):", tau_w_Re_335666)

# Find the Boundary layer thickness for CASE 4
delta_Re_540600 = np.interp(0.99, u_Re_540600/Uinf_Re_540600, y_Re_540600)
print("BL thickness Case4_Re_540600 (mm):", np.round(delta_Re_540600, 3))

# Calculate the gradient at the wall for wall-shear-stress
tau_w_Re_540600 = 1000 * mu_air * \
    (u_Re_540600[1] - u_Re_540600[0]) / (y_Re_540600[1] - y_Re_540600[0])
print(" wall shear stress Case3_Re_540600 (Pa):", tau_w_Re_540600)

# Find the Boundary layer thickness for CASE 5
delta_Re_441666 = np.interp(0.99, u_Re_441666/Uinf_Re_441666, y_Re_441666)
print("BL thickness Case5_Re_441666 (mm):", np.round(delta_Re_441666, 3))

# Calculate the gradient at the wall for wall-shear-stress
tau_w_Re_441666 = 1000 * mu_air * \
    (u_Re_441666[1] - u_Re_441666[0]) / (y_Re_441666[1] - y_Re_441666[0])
print("wall shear stress Case3_Re_441666 (Pa):", tau_w_Re_441666)




# Plot the velocity profiles in non-dimensional units
plt.figure(figsize=(8, 6))
plt.plot(u_Re_261886 / u_Re_261886.max(), y_Re_261886 / delta_Re_261886, 'o-', label='Re = 261886')
plt.plot(u_Re_194333 / u_Re_194333.max(), y_Re_194333 / delta_Re_194333, 's-', label='Re = 194333')
plt.plot(u_Re_335666 / u_Re_335666.max(), y_Re_335666 / delta_Re_335666, 'p-', label='Re = 335666')
plt.plot(u_Re_540600 / u_Re_540600.max(), y_Re_540600 / delta_Re_540600, 'h-', label='Re = 540600')
plt.plot(u_Re_441666 / u_Re_441666.max(), y_Re_441666 / delta_Re_441666, 'h-', label='Re = 441666')

# Adding labels, title, legend, and grid
plt.xlabel("U/U_inf", fontsize=12)
plt.ylabel("y/delta", fontsize=12)
plt.title("Non-dimensionalised Profiles for Different Reynolds Numbers", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

def velocity_profile(y_by_delta, n):
    # Theoretical profile function: (y/delta)^(1/n)
    return (y_by_delta)**(1/n)

# Perform curve fitting for each Reynolds number
def fit_and_plot_distinct(y, u, delta, u_inf, label, color):
    # Normalize the data
    y_by_delta = y / delta
    u_by_u_inf = u / u_inf

    # Limit data range to y/delta <= 1
    valid_indices = y_by_delta <= 1
    y_by_delta = y_by_delta[valid_indices]
    u_by_u_inf = u_by_u_inf[valid_indices]

    # Fit the data
    popt, _ = curve_fit(velocity_profile, y_by_delta, u_by_u_inf, p0=[5], bounds=(1, [20]))
    n_fitted = popt[0]

    # Plot the fitted curve
    y_fit = np.linspace(0, 1, 100)
    u_fit = velocity_profile(y_fit, n_fitted)
    plt.plot(y_by_delta, u_by_u_inf, 'o', color=color, label=f'{label} (data)')
    plt.plot(y_fit, u_fit, '-', color=color, linewidth=2, label=f'{label} (fit, n={n_fitted:.2f})')

    return n_fitted

# Define a distinct color palette
colors = plt.cm.tab10.colors

# Sort Reynolds numbers for proper order
reynolds_cases = [
    ('Re = 194333', y_Re_194333, u_Re_194333, delta_Re_194333, Uinf_Re_194333, colors[0]),
    ('Re = 261886', y_Re_261886, u_Re_261886, delta_Re_261886, Uinf_Re_261886, colors[1]),
    ('Re = 335666', y_Re_335666, u_Re_335666, delta_Re_335666, Uinf_Re_335666, colors[2]),
    ('Re = 441666', y_Re_441666, u_Re_441666, delta_Re_441666, Uinf_Re_441666, colors[3]),
    ('Re = 540600', y_Re_540600, u_Re_540600, delta_Re_540600, Uinf_Re_540600, colors[4]),
]

# Initialize the plot
plt.figure(figsize=(10, 8))
n_values = {}

# Loop through sorted Reynolds cases and plot them
for label, y, u, delta, Uinf, color in reynolds_cases:
    n_values[label] = fit_and_plot_distinct(y, u, delta, Uinf, label, color)

# Finalize the plot
plt.xlabel("y/delta", fontsize=12)
plt.ylabel("U/U_inf", fontsize=12)
plt.title("Fitted Exponential Velocity Profiles for Turbulent Boundary Layer", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Print the fitted n-values in the correct order
for key, value in n_values.items():
    print(f"{key}: n = {value:.2f}")



# Code to calculate the total viscous drag coefficient (C_F) for each case

"""This function computes all boundary layer properties including
displacement and momentum thicknesses as well 
as shape factor, can be used for each case"""

def BLprop(u,y):
    Uinf = u.max()
    delta = np.interp(0.99,u/Uinf,y)
    tau_w = 1000*mu_air*(u[1]-u[0])/(y[1]-y[0])

    dispt = np.trapz(1-(u/u.max()),y)
    momt = np.trapz(u/u.max()*(1-(u/u.max())),y)
    H = dispt/momt

    return [Uinf, delta, tau_w, dispt, momt, H]

# CASE 1
[uinf_Re_261886,delta_Re_261886,tau_w_Re_261886,dispt_Re_261886,momt_Re_261886,H_Re_261886] = BLprop(u_Re_261886,y_Re_261886)
# CASE 2
[uinf_Re_194333,delta_Re_194333,tau_w_Re_194333,dispt_Re_194333,momt_Re_194333,H_Re_194333] = BLprop(u_Re_194333,y_Re_194333)
# CASE 3
[uinf_Re_335666,delta_Re_335666,tau_w_Re_335666,dispt_Re_335666,momt_Re_335666,H_Re_335666] = BLprop(u_Re_335666,y_Re_335666)
# CASE 4
[uinf_Re_540600,delta_Re_540600,tau_w_Re_540600,dispt_Re_540600,momt_Re_540600,H_Re_540600] = BLprop(u_Re_540600,y_Re_540600)
# CASE 5
[uinf_Re_441666,delta_Re_441666,tau_w_Re_441666,dispt_Re_441666,momt_Re_441666,H_Re_441666] = BLprop(u_Re_441666,y_Re_441666)
# CASE 6 
[uinf_Re_192566,delta_Re_192566,tau_w_Re_192566,dispt_Re_192566,momt_Re_192566,H_Re_192566] = BLprop(u_Re_192566,y_Re_192566)
# CASE 7 
[uinf_Re_602786,delta_Re_602786,tau_w_Re_602786,dispt_Re_602786,momt_Re_602786,H_Re_602786] = BLprop(u_Re_602786,y_Re_602786)
# CASE 8
[uinf_Re_462866,delta_Re_462866,tau_w_Re_462866,dispt_Re_462866,momt_Re_462866,H_Re_462866] = BLprop(u_Re_462866,y_Re_462866)
# CASE 9
[uinf_Re_143100,delta_Re_143100,tau_w_Re_143100,dispt_Re_143100,momt_Re_143100,H_Re_143100] = BLprop(u_Re_143100,y_Re_143100)
# CASE 10
[uinf_Re_509683,delta_Re_509683,tau_w_Re_509683,dispt_Re_509683,momt_Re_509683,H_Re_509683] = BLprop(u_Re_509683,y_Re_509683)
# CASE 11
[uinf_Re_538833,delta_Re_538833,tau_w_Re_538833,dispt_Re_538833,momt_Re_538833,H_Re_538833] = BLprop(u_Re_538833,y_Re_538833)
# CASE 12
[uinf_Re_415166,delta_Re_415166,tau_w_Re_415166,dispt_Re_415166,momt_Re_415166,H_Re_415166] = BLprop(u_Re_415166,y_Re_415166)
# CASE 13
[uinf_Re_215886,delta_Re_215886,tau_w_Re_215886,dispt_Re_215886,momt_Re_215886,H_Re_215886] = BLprop(u_Re_215886,y_Re_215886)
# CASE 14
[uinf_Re_503500,delta_Re_503500,tau_w_Re_503500,dispt_Re_503500,momt_Re_503500,H_Re_503500] = BLprop(u_Re_503500,y_Re_503500)
# CASE 15
[uinf_Re_226133,delta_Re_226133,tau_w_Re_226133,dispt_Re_226133,momt_Re_226133,H_Re_226133] = BLprop(u_Re_226133,y_Re_226133)
# CASE 16
[uinf_Re_348033,delta_Re_348033,tau_w_Re_348033,dispt_Re_348033,momt_Re_348033,H_Re_348033] = BLprop(u_Re_348033,y_Re_348033)


# This code compute the total viscous drag of a flat plate of length 0.265m
L = 0.265  # m
CF_case1 = 2*0.001*momt_Re_261886/L
CF_case2 = 2*0.001*momt_Re_194333/L
CF_case3 = 2*0.001*momt_Re_335666/L
CF_case4 = 2*0.001*momt_Re_540600/L
CF_case5 = 2*0.001*momt_Re_441666/L
CF_case6 = 2*0.001*momt_Re_192566/L
CF_case7 = 2*0.001*momt_Re_602786/L
CF_case8 = 2*0.001*momt_Re_462866/L
CF_case9 = 2*0.001*momt_Re_143100/L
CF_case10 = 2*0.001*momt_Re_509683/L
CF_case11 = 2*0.001*momt_Re_538833/L
CF_case12 = 2*0.001*momt_Re_415166/L
CF_case13 = 2*0.001*momt_Re_215886/L
CF_case14 = 2*0.001*momt_Re_503500/L
CF_case15 = 2*0.001*momt_Re_226133/L
CF_case16 = 2*0.001*momt_Re_348033/L

print("Total momentum thickness CF_case1 = ",momt_Re_261886)
print("Total momentum thickness CF_case2 = ",momt_Re_194333)
print("Total momentum thickness CF_case3 = ",momt_Re_335666)
print("Total momentum thickness CF_case4 = ",momt_Re_540600)
print("Total momentum thickness CF_case5 = ",momt_Re_441666)
print("Total momentum thickness CF_case6 = ",momt_Re_192566)
print("Total momentum thickness CF_case7 = ",momt_Re_602786)
print("Total momentum thickness CF_case8 = ",momt_Re_462866)
print("Total momentum thickness CF_case9 = ",momt_Re_143100)
print("Total momentum thickness CF_case10 = ",momt_Re_509683)
print("Total momentum thickness CF_case11 = ",momt_Re_538833)
print("Total momentum thickness CF_case12 = ",momt_Re_415166)
print("Total momentum thickness CF_case13 = ",momt_Re_215886)
print("Total momentum thickness CF_case14 = ",momt_Re_503500)
print("Total momentum thickness CF_case15 = ",momt_Re_226133)
print("Total momentum thickness CF_case16 = ",momt_Re_348033)



print("Total viscous drag coefficient CF_case1 = ",2*CF_case1)
print("Total viscous drag coefficient CF_case2 = ",2*CF_case2)
print("Total viscous drag coefficient CF_case3 = ",2*CF_case3)
print("Total viscous drag coefficient CF_case4 = ",2*CF_case4)
print("Tota  viscous drag coefficient CF_case5 = ",2*CF_case5)

print("H_Re_261886 - H = ",np.round(H_Re_261886,4))
print("H_Re_194333 - H = ",np.round(H_Re_194333,4))
print("H_Re_335666 - H = ",np.round(H_Re_335666,4))
print("H_Re_540600 - H = ",np.round(H_Re_540600,4))
print(" H_Re_441666 - H = ",np.round(H_Re_441666,4))

# Given Reynolds numbers and CF 
Re_values = np.array([
    261886, 194333, 335666, 540600, 441666, 
    192566, 602786, 462866, 143100, 509683, 
    538833, 415166, 215886, 503500, 226133, 348033
])

CF_values = np.array([
    CF_case1, CF_case2, CF_case3, CF_case4, CF_case5,
    CF_case6, CF_case7, CF_case8, CF_case9, CF_case10,
    CF_case11, CF_case12, CF_case13, CF_case14, CF_case15, CF_case16
])

# Define the power law function
def func(Rec, A, b):
    return A * (Rec ** b)

# Filter data to exclude outliers
# criterion: exclude points where 
mask = Re_values >= 200000
Re_filtered = Re_values[mask]
CF_filtered = CF_values[mask]

# Perform the curve fit on filtered data
params, _ = curve_fit(func, Re_filtered, CF_filtered)
A, b = params

# Print the fitted parameters
print(f"Fitted Parameters (Filtered Data):\nA = {A:.6f}\nb = {b:.6f}")

# Generate the fitted curve
Re_fit = np.linspace(min(Re_values), max(Re_values), 500)
CF_fit = func(Re_fit, A, b)

# Plotting
plt.figure(figsize=(8, 6))

# Plot CF values (red crosses)
plt.plot(Re_values, CF_values, 'rx', label="CF Data Points", markersize=8)

# Plot empirical curve (optional comparison)
Re_empirical = np.linspace(min(Re_values), max(Re_values), 500)
CF_empirical = 0.074 / (Re_empirical ** 0.2)
plt.plot(Re_empirical, CF_empirical, 'b-', label="Empirical CF = 0.074 / Re^0.2", linewidth=2)

# Plot fitted curve
plt.plot(Re_fit, CF_fit, 'g--', label=f"Fitted Curve (Filtered): CF = {A:.4f} * Re^{b:.4f}", linewidth=2)

# Plot settings
# Highlight excluded points in black circles
plt.plot(Re_values[~mask], CF_values[~mask], 'ko', label="Excluded Points", markerfacecolor='none' ,markersize=10)

plt.xlabel("Reynolds Number (Re)", fontsize=12)
plt.ylabel("Total Viscous Drag Coefficient (CF)", fontsize=12)
plt.title("Total CF vs Reynolds Number", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.show()


