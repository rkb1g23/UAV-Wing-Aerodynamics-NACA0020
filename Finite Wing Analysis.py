# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 00:09:25 2024

@author: rkuma
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Given values
T = 298  # Temperature in Kelvin
P = 101325  # Pressure in Pascal
R = 287  # Specific gas constant for air in J/(kg·K)

# Calculate density
rho = P / (R * T)

# Print the result
print(f"Density of air: {rho:.2f} kg/m³")


# Note down the main characteristics of the wing and the flow

Uinf = 20  # m/s (flow velocity)
# m (This is the half-span of the wing since we only make measurements of half-wing)
b = 1.11
c = 0.4  # m (chord)
bf = 2*b  # This is the total span of the wing to account for two wing-tip vortices

# This is the wing area. Note that the wing in the experiment is only done half of the wing.
S = bf*c

AR = bf*bf/S  # This is the aspect ratio of the wing.


# Define the range of angles of attack measured in the lab
AoAdegrees = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
              8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Convert AoAdegrees to radians
# This converts angle of attack to radians
AoA = np.array(AoAdegrees) * np.pi / 180

# Define the mean values for the side forces
S = [
    -32.434, -25.424, -18.744, -11.696, -4.709, 2.923, 10.139, 16.985, 24.071,
    30.967, 37.709, 44.154, 50.946, 57.432, 63.566, 69.096, 74.801, 80.632,
    86.536, 91.3, 94.867, 98.226, 100.557, 102.493, 104.732
]

# Define the mean values for the drag forces
D = [
    -1.995, -2.666, -3.086, -3.337, -3.409, -3.374, -3.133, -2.722, -2.099,
    -1.344, -0.36, 0.679, 2.041, 3.465, 5.024, 6.539, 8.206, 10.066, 12.017,
    13.81, 15.483, 16.868, 17.849, 18.674, 19.314
]

# Create a DataFrame from the arrays
data = {
    'AoA (radians)': AoA,
    'S': S,
    'D': D
}
df = pd.DataFrame(data)

# Define the path where you want to save the CSV file
csv_file_path = 'output_finite_wing.csv'  # Replace with your desired file path

# Write the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
print(f"Data has been written to {csv_file_path}")


# Now that you have your true lift and drag values, compute  𝐶𝐿, 𝐶𝐷


# Calculate dynamic pressure
q_infinity = 0.5 * rho * Uinf**2
print(f"Dynamic pressure (q_inf): {q_infinity:.2f} Pa")

# Step 4: Reference area (using half-span only)
A_ref = b * c  # Reference area in m²
print(f"Reference area (A_ref): {A_ref:.2f} m²")

# Step 5: Read the CSV file with AoA, S, and D
# Replace with your input file path
input_csv_file = r"C:\Users\rkuma\Downloads\output_finite_wing.csv"
df = pd.read_csv(input_csv_file)

# Extract AoA, S, and D from the DataFrame
AoA = df['AoA (radians)'].values
S = df['S'].values
D = df['D'].values


# Step 6: Compute true lift (L) and drag (D)
L = S * np.cos(AoA) - D * np.sin(AoA)
D_true = D * np.cos(AoA) + S * np.sin(AoA)

# Step 7: Compute C_L and C_D
C_L = L / (q_infinity * A_ref)
C_D = D_true / (q_infinity * A_ref)

# Add AoA in degrees for clarity
AoA_degrees = np.degrees(AoA)

# Step 8: Create a new DataFrame for output
output_data = {
    'AoA (degrees)': AoA_degrees,
    'C_L': C_L,
    'C_D': C_D
}
output_df = pd.DataFrame(output_data)

# Step 9: Write the output DataFrame to a new CSV file
output_csv_file = 'output_CL_CD.csv'  # Replace with your desired file path
output_df.to_csv(output_csv_file, index=False)
print(f"Computed C_L and C_D have been written to {output_csv_file}")
