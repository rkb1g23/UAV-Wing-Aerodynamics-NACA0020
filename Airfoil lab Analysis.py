# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:20:40 2024

@author: rkuma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def Aerofoil(x):
    '''This function takes the x coordinate and returns the positive y coordinate of the aerofoil'''
    t=0.20
    y = (5*t)*((0.2969*x**0.5)-(0.1260*x)-(0.3516*x**2)+(0.2843*x**3)-(0.1015*x**4))
    return(y)

#This is set up so if the aerofoil is turned clockwise the upper surface with have the tapping positons below otherwise
#just switch upper and lower surface around
x = [1, 2, 4.5, 7.5, 11, 14.5, 20, 26, 32.1, 38, 44, 50]
xc_upper= []
xc_lower =[]
for i in range(12):
    if i % 2 == 0:
        xc_lower.append(x[i]/63)
    if i % 2 == 1:
        xc_upper.append(x[i]/63)
        
plt.plot([Aerofoil(i) for i in xc_upper], xc_upper, 'ro', label= 'Upper Surface Tappings')
plt.plot([Aerofoil(i) for i in np.linspace(0,1,100)], np.linspace(0,1,100), 'g-')
plt.plot([-Aerofoil(i) for i in xc_lower], xc_lower, 'bo', label = 'Lower Surface Tappings')
plt.plot([-Aerofoil(i) for i in np.linspace(0,1,100)], np.linspace(0,1,100), 'g-')
plt.gca().invert_yaxis()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.axis('scaled')
plt.show()

# Given values
Temperature_deg = 24  # degrees C
Pressure = 102500  # Pa
R = 287.05  # J/(kg·K)

# Convert temperature to Kelvin
Temperature_kelvin = Temperature_deg + 273

# Calculate the density using the ideal gas law formula
rho = Pressure / (R * Temperature_kelvin)


negative_Four_data = pd.read_csv(r"C:\Users\rkuma\Downloads\-4deg.csv")
Variable, Manometer = negative_Four_data.columns
Variable_negativeFour = negative_Four_data[Variable]
Manometer_negativeFour = negative_Four_data[Manometer]

negative_Four_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\-4_deg.csv")
Variable, Manometer = negative_Four_data2.columns
Variable_negativeFour2 = negative_Four_data2[Variable]
Manometer_negativeFour2 = negative_Four_data2[Manometer]

negative_Four_data3 = pd.read_csv(r"C:\Users\rkuma\Downloads\-4_deg (1).csv")
Variable, Manometer = negative_Four_data3.columns
Variable_negativeFour3 = negative_Four_data3[Variable]
Manometer_negativeFour3 = negative_Four_data3[Manometer]

zero_degree_data = pd.read_csv(r"C:\Users\rkuma\Downloads\0 degrees.csv")
Variable, Manometer = zero_degree_data.columns
Variable_zero = zero_degree_data[Variable]
Manometer_zero = zero_degree_data[Manometer]

zero_degree_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\0_deg.csv")
Variable, Manometer = zero_degree_data2.columns
Variable_zero2 = zero_degree_data2[Variable]
Manometer_zero2 = zero_degree_data2[Manometer]

Four_data = pd.read_csv(r"C:\Users\rkuma\Downloads\4_deg (1).csv")
Variable, Manometer = Four_data.columns
Variable_Four = Four_data[Variable]
Manometer_Four = Four_data[Manometer]

Four_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\4 degrees.csv")
Variable, Manometer = Four_data2.columns
Variable_Four2 = Four_data2[Variable]
Manometer_Four2 = Four_data2[Manometer]

Eight_data = pd.read_csv(r"C:\Users\rkuma\Downloads\8_deg.csv")
Variable, Manometer = Eight_data.columns
Variable_Eight = Eight_data[Variable]
Manometer_Eight = Eight_data[Manometer]

Eight_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\8_deg (1).csv")
Variable, Manometer = Eight_data2.columns
Variable_Eight2 = Eight_data2[Variable]
Manometer_Eight2 = Eight_data2[Manometer]

twelve_data = pd.read_csv(r"C:\Users\rkuma\Downloads\12_deg.csv")
Variable, Manometer = twelve_data.columns
Variable_twelve = twelve_data[Variable]
Manometer_twelve = twelve_data[Manometer]

twelve_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\12 degrees.csv")
Variable, Manometer = twelve_data2.columns
Variable_twelve2 = twelve_data2[Variable]
Manometer_twelve2 = twelve_data2[Manometer]

Fourteen_data = pd.read_csv(r"C:\Users\rkuma\Downloads\14_deg.csv")
Variable, Manometer = Fourteen_data.columns
Variable_Fourteen = Fourteen_data[Variable]
Manometer_Fourteen = Fourteen_data[Manometer]

Fourteen_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\14deg.csv")
Variable, Manometer = Fourteen_data2.columns
Variable_Fourteen2 = Fourteen_data2[Variable]
Manometer_Fourteen2 = Fourteen_data2[Manometer]



# Function for the CP
def Cp(P, atm, Vel):
    deltap = (P-atm)*1000*9.81
    return(deltap/(0.5*Vel**2 * rho))

# Function for the velocity

def Velocity(tot, free):
    deltap = (tot - free)*1000*9.81
    vel = ((deltap)/(0.5*rho))**0.5
    return(vel)


# Define a function to process the array
def pressuredistribution(dataarray):
    Vel = Velocity(dataarray[0], dataarray[1])
    Cparray = [Cp(dataarray[i], dataarray[1], Vel) for i in range(2, 14)]
    CpLower = Cparray[0:6]
    CpUpper = Cparray[6:12]
    return(Vel, CpUpper, CpLower)

def grad(x):
    t = 0.20
    y = ((0.5 * 0.2969 * x**(-0.5)) - (0.1260) - (2 * 0.3516 * x) + (3 * 0.2843 * x**2) - (4 * 0.1015 * x**3))
    return(y)

def clcd(cpu, cpl, aoa):
    aoa = np.radians(aoa)
    cn = np.trapz(cpl, xc_lower) - np.trapz(cpu, xc_upper)
    cpgradl = [cpl[i] * grad(xc_lower[i]) for i in range(len(xc_lower))]
    cpgradu = [cpu[i] * grad(xc_upper[i]) for i in range(len(xc_upper))]
    ca = np.trapz(cpgradu, xc_upper) - np.trapz(cpgradl, xc_lower)
    cl = (cn * np.cos(aoa) - ca * np.sin(aoa))
    cd = (ca * np.cos(aoa) + cn * np.sin(aoa))
    return(cl, cd)

# Calculate pressure, lift, drag for AoA=-4 deg
Vel, Cpu_negativefour, Cpl_negativefour = pressuredistribution(Manometer_negativeFour)
cl_negativefour, cd_negativefour = clcd(Cpu_negativefour, Cpl_negativefour, -4)

Cpu_negativefour = [i * -1 for i in Cpu_negativefour]
Cpl_negativefour = [i * -1 for i in Cpl_negativefour]

# Calculate pressure, lift, drag for AoA=-4 deg
Vel, Cpu_negativefour2, Cpl_negativefour2 = pressuredistribution(Manometer_negativeFour2)
cl_negativefour2, cd_negativefour2 = clcd(Cpu_negativefour2, Cpl_negativefour2, -4)

Cpu_negativefour2 = [i * -1 for i in Cpu_negativefour2]
Cpl_negativefour2 = [i * -1 for i in Cpl_negativefour2]

# Calculate pressure, lift, drag for AoA=-4 deg
Vel, Cpu_negativefour3, Cpl_negativefour3 = pressuredistribution(Manometer_negativeFour3)
cl_negativefour3, cd_negativefour3 = clcd(Cpu_negativefour3, Cpl_negativefour3, -4)

Cpu_negativefour3 = [i * -1 for i in Cpu_negativefour3]
Cpl_negativefour3 = [i * -1 for i in Cpl_negativefour3]

# Calculate pressure, lift, drag for AoA= 0 deg
Vel, Cpu_zero, Cpl_zero = pressuredistribution(Manometer_zero)
cl_zero, cd_zero = clcd(Cpu_zero, Cpl_zero, 0)

Cpu_zero = [i * -1 for i in Cpu_zero]
Cpl_zero = [i * -1 for i in Cpl_zero]

# Calculate pressure, lift, drag for AoA= 0 deg
Vel, Cpu_zero2, Cpl_zero2 = pressuredistribution(Manometer_zero2)
cl_zero2, cd_zero2 = clcd(Cpu_zero2, Cpl_zero2, 0)

Cpu_zero2 = [i * -1 for i in Cpu_zero2]
Cpl_zero2 = [i * -1 for i in Cpl_zero2]

# Calculate pressure, lift, drag for AoA=4 deg
Vel, Cpu_four, Cpl_four = pressuredistribution(Manometer_Four)
cl_four, cd_four = clcd(Cpu_four, Cpl_four, 4)

Cpu_four = [i * -1 for i in Cpu_four]
Cpl_four = [i * -1 for i in Cpl_four]

# Calculate pressure, lift, drag for AoA=4 deg
Vel, Cpu_four2, Cpl_four2 = pressuredistribution(Manometer_Four2)
cl_four2, cd_four2 = clcd(Cpu_four2, Cpl_four2, 4)

Cpu_four2 = [i * -1 for i in Cpu_four2]
Cpl_four2 = [i * -1 for i in Cpl_four2]

# Calculate pressure, lift, drag for AoA=8 deg
Vel, Cpu_eight, Cpl_eight = pressuredistribution(Manometer_Eight)
cl_eight, cd_eight = clcd(Cpu_eight, Cpl_eight, 8)

Cpu_eight = [i * -1 for i in Cpu_eight]
Cpl_eight = [i * -1 for i in Cpl_eight]

# Calculate pressure, lift, drag for AoA=8 deg
Vel, Cpu_eight2, Cpl_eight2 = pressuredistribution(Manometer_Eight2)
cl_eight2, cd_eight2 = clcd(Cpu_eight2, Cpl_eight2, 8)

Cpu_eight2 = [i * -1 for i in Cpu_eight2]
Cpl_eight2 = [i * -1 for i in Cpl_eight2]

# Calculate pressure, lift, drag for AoA= 12 deg
Vel, Cpu_twelve, Cpl_twelve = pressuredistribution(Manometer_twelve)
cl_twelve, cd_twelve = clcd(Cpu_twelve, Cpl_twelve, 12)

Cpu_twelve = [i * -1 for i in Cpu_twelve]
Cpl_twelve = [i * -1 for i in Cpl_twelve]

# Calculate pressure, lift, drag for AoA= 12 deg
Vel, Cpu_twelve2, Cpl_twelve2 = pressuredistribution(Manometer_twelve2)
cl_twelve2, cd_twelve2 = clcd(Cpu_twelve2, Cpl_twelve2, 12)

Cpu_twelve2 = [i * -1 for i in Cpu_twelve2]
Cpl_twelve2 = [i * -1 for i in Cpl_twelve2]

# Calculate pressure, lift, drag for AoA=14 deg
Vel, Cpu_fourteen, Cpl_fourteen = pressuredistribution(Manometer_Fourteen)
cl_fourteen, cd_fourteen = clcd(Cpu_fourteen, Cpl_fourteen, 14)

Cpu_fourteen = [i * -1 for i in Cpu_fourteen]
Cpl_fourteen = [i * -1 for i in Cpl_fourteen]

# Calculate pressure, lift, drag for AoA=14 deg
Vel, Cpu_fourteen2, Cpl_fourteen2 = pressuredistribution(Manometer_Fourteen2)
cl_fourteen2, cd_fourteen2 = clcd(Cpu_fourteen2, Cpl_fourteen2, 14)

Cpu_fourteen2 = [i * -1 for i in Cpu_fourteen2]
Cpl_fourteen2 = [i * -1 for i in Cpl_fourteen2]


# CFD RESULTS FOR 4ª
Pressure_CFD_Four = pd.read_csv(r"C:\Users\rkuma\Downloads\cfd_four.csv")

xx, CFD = Pressure_CFD_Four.columns
xx = Pressure_CFD_Four[xx]
CFD1 = Pressure_CFD_Four[CFD]

# CFD RESULTS FOR 8ª
Pressure_CFD_Eight = pd.read_csv(r"C:\Users\rkuma\Downloads\cfd_eight.csv")
xx2, CFD2 = Pressure_CFD_Eight.columns
xx2 = Pressure_CFD_Eight[xx2]
CFD2 = Pressure_CFD_Eight[CFD2]


# CFD RESULTS FOR 14ª
Pressure_CFD_14 = pd.read_csv(r"C:\Users\rkuma\Downloads\cfd_eight.csv")
xx3, CFD3 = Pressure_CFD_14.columns
xx3 = Pressure_CFD_14[xx3]
CFD3 = Pressure_CFD_14[CFD3]


# XFOIL data 4º
xfoil_data = pd.read_csv(r"C:\Users\rkuma\Downloads\xfoil_results.csv")

x1, cp1 = xfoil_data.columns

# Extract x/c and Cp
x_xfoil = xfoil_data[x1]  # x/c
cp_xfoil = xfoil_data[cp1]  # Cp

# XFOIL data 8º
xfoil_data8 = pd.read_csv(r"C:\Users\rkuma\Downloads\xfoil_eight.csv")

x2, cp2 = xfoil_data8.columns

# Extract x/c and Cp
x_xfoil2 = xfoil_data8[x2]  # x/c
cp_xfoil2 = xfoil_data8[cp2]  # Cp

# XFOIL data 14º
xfoil_data14 = pd.read_csv(r"C:\Users\rkuma\Downloads\cpplot_14.csv")

x3, cp3 = xfoil_data14.columns

# Extract x/c and Cp
x_xfoil3 = xfoil_data14[x3]  # x/c
cp_xfoil3 = xfoil_data14[cp3]  # Cp








# Define a larger figure size for better spacing
figure_size = (10, 8)

# 4° AoA
plt.figure(figsize=figure_size)
plt.title("Pressure Distribution at 4° AoA", fontsize=18, fontweight='bold')

# Use solid lines for CFD and XFOIL, markers for experimental data
plt.plot(xx/max(xx), -CFD1, label="CFD", linewidth=2, color='blue')
plt.plot(x_xfoil, -cp_xfoil, label="XFOIL", linestyle='-', linewidth=2, color='orange')
plt.plot(xc_lower, Cpl_four, 'o--', label="Lower Surface (Exp)", color='green', markersize=8)
plt.plot(xc_upper, Cpu_four, 'x--', label="Upper Surface (Exp)", color='red', markersize=8)

# Customizations
plt.legend(loc='upper right', fontsize=14, frameon=True, edgecolor='black')
plt.xlabel('x/c', fontsize=16)
plt.ylabel('-Cp', fontsize=16)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 8° AoA
plt.figure(figsize=figure_size)
plt.title("Pressure Distribution at 8° AoA", fontsize=18, fontweight='bold')
plt.plot(xx2/max(xx2), -CFD2, label="CFD", linewidth=2, color='blue')
plt.plot(x_xfoil2, -cp_xfoil2, label="XFOIL", linestyle='-', linewidth=2, color='orange')
plt.plot(xc_lower, Cpl_eight, 'o--', label="Lower Surface (Exp)", color='green', markersize=8)
plt.plot(xc_upper, Cpu_eight, 'x--', label="Upper Surface (Exp)", color='red', markersize=8)
plt.legend(loc='upper right', fontsize=14, frameon=True, edgecolor='black')
plt.xlabel('x/c', fontsize=16)
plt.ylabel('-Cp', fontsize=16)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 14° AoA
plt.figure(figsize=figure_size)
plt.title("Pressure Distribution at 14° AoA", fontsize=18, fontweight='bold')
plt.plot(xx3/max(xx3), -CFD3, label="CFD", linewidth=2, color='blue')
plt.plot(x_xfoil3, -cp_xfoil3, label="XFOIL", linestyle='-', linewidth=2, color='orange')
plt.plot(xc_lower, Cpl_fourteen, 'o--', label="Lower Surface (Exp)", color='green', markersize=8)
plt.plot(xc_upper, Cpu_fourteen, 'x--', label="Upper Surface (Exp)", color='red', markersize=8)
plt.legend(loc='upper right', fontsize=14, frameon=True, edgecolor='black')
plt.xlabel('x/c', fontsize=16)
plt.ylabel('-Cp', fontsize=16)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Show all plots
plt.show()

#  C_L vs AoA Analysis

# Given data: Angles of attack and respective Cl values
# Angles of attack in degrees
angles_of_attack_degrees = [-4, 0, 4, 8, 12, 14]
# Corresponding Cl values
cl_values = [
    cl_negativefour,  # For -4 degrees
    cl_zero2,                  # For 0 degrees
    cl_four,                   # For 4 degrees
    cl_eight,                 # For 8 degrees
    cl_twelve,               # For 12 degrees
    cl_fourteen,          # For 14 degrees
]

print(cl_values)
# Convert angles to radians for fitting
angles_of_attack_radians = np.radians(angles_of_attack_degrees)

# Define linear fit function
def linear_fit(x, m, c):
    return m * x + c

# Perform curve fitting
params, _ = curve_fit(linear_fit, angles_of_attack_radians, cl_values)
fitted_gradient, fitted_intercept = params

# Theoretical calculation (assuming alpha_0 = 0 for simplicity)
alpha_0 = 0  # Zero-lift angle of attack in radians
theoretical_cl = 2 * np.pi * (angles_of_attack_radians - alpha_0)

# XFOIL data 

XFOIL_data = pd.read_csv(r"C:\Users\rkuma\Downloads\naca0020xfoil.csv")

# Extract data from the CSV file
xfoil_angles_of_attack = XFOIL_data["alpha"]
xfoil_cl_values = XFOIL_data["CL"]

XFOIL_data2 = pd.read_csv(r"C:\Users\rkuma\Downloads\invsicid.csv")

# Extract data from the CSV file
xfoil_angles_of_attack = XFOIL_data2["alpha"]
xfoil_cl_values2 = XFOIL_data2["CL"]




# CFD CL_ANALYSIS

CFD_CL_data = pd.read_csv(r"C:\Users\rkuma\Downloads\C_L_CFD.csv")
# Extract data from the CSV file
ANSYS_angles_of_attack = CFD_CL_data["AoA"]
ANSYS_CL_values = CFD_CL_data["CL_CFD"]

# Filter the CFD data to only include angles of attack up to 10 degrees
linear_region_mask = ANSYS_angles_of_attack <= 10
linear_angles_of_attack = ANSYS_angles_of_attack[linear_region_mask]
linear_cl_values = ANSYS_CL_values[linear_region_mask]

# Convert filtered angles to radians for fitting
linear_angles_of_attack_radians = np.radians(linear_angles_of_attack)

# Perform curve fitting on the linear region to find the lift curve slope
linear_params, linear_covariance = curve_fit(linear_fit, linear_angles_of_attack_radians, linear_cl_values)
linear_gradient, linear_intercept = linear_params

# Generate fitted values for the linear region
linear_fitted_cl_values = linear_fit(linear_angles_of_attack_radians, *linear_params)

filtered_angles_of_attack = ANSYS_angles_of_attack[linear_region_mask]
filtered_cl_values = ANSYS_CL_values[linear_region_mask]







# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(angles_of_attack_degrees, cl_values, color="blue", label="Computed Cl values")
plt.plot(angles_of_attack_degrees, linear_fit(np.radians(angles_of_attack_degrees), *params), 
         color="red", linestyle="--", label=f"Fitted Line: Gradient = {fitted_gradient:.2f}")
# Plot the theoretical line
plt.plot(np.degrees(angles_of_attack_radians), theoretical_cl, 
         color="green", linestyle="-", label="Theoretical Line (2πα)")
plt.plot(xfoil_angles_of_attack, xfoil_cl_values, 
         color="blue", linestyle="-", label="XFOIL viscous Line ")
plt.plot(xfoil_angles_of_attack, xfoil_cl_values2, 
         color="magenta", linestyle="-", label="XFOIL inviscid Line ")

plt.plot(filtered_angles_of_attack, linear_fitted_cl_values, 
         color="red", linestyle="-", label=f"ANSYS (Linear Region): Gradient = {linear_gradient:.2f}")




# Annotate the plot
plt.title("Cl vs Angle of Attack")
plt.xlabel("Angle of Attack (degrees)")
plt.ylabel("Lift Coefficient (Cl)")
plt.legend()
plt.grid()
plt.show()

# Print the fitted gradient and intercept
print(f"Fitted Gradient (in radians): {fitted_gradient:.2f}")







    


    


    








