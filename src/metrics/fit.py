import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
# df = pd.read_csv('input/use_hp_combined.csv')
df = pd.read_csv('input/carbon_specs_all_use.csv')

# Check for NaN values and handle them
print(f"Initial shape of dataset: {df.shape}")
df.dropna(subset=['Power', 'Use Carbon Footprint'], inplace=True)  # Drop rows where NaN values exist in 'Power' or 'Use Carbon Footprint'
print(f"Shape after dropping NaN values: {df.shape}")

x = df['Electricity'].values.reshape(-1, 1)  # Reshape for sklearn
y = df['Use Carbon Footprint'].values

# Perform linear regression
model = LinearRegression()
model.fit(x, y)

# Get the slope (k) and the intercept (b)
slope = model.coef_[0]
intercept = model.intercept_

print(f'Slope (k): {slope}')
print(f'Intercept (b): {intercept}')

# Generate predictions for the fitting line
x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_fit = model.predict(x_fit)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual data', color='blue')
plt.plot(x_fit, y_fit, label='Fitting line', color='red')

# plt.xlabel('Power Supply (W)')
plt.xlabel('Electricity (kWh)')
plt.ylabel('Use Carbon Footprint (kgCO2eq.)')
# plt.title('Linear Fit of Use Carbon Footprint vs. Power Supply')
# plt.legend()
plt.savefig('output/elec_linear_fit.png')
