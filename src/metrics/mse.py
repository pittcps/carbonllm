import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the data
hp_combined_path = 'hp_combined.csv'
act_carbon_output_path = 'total_carbon_output3.csv'
hp_combined_data = pd.read_csv(hp_combined_path)
act_carbon_output_data = pd.read_csv(act_carbon_output_path)

# Merge the datasets on 'Commercial Name'
merged_data = pd.merge(hp_combined_data, act_carbon_output_data, on='Commercial Name')

# Replace NaN values with 0 in the relevant columns
merged_data['Manufacturing2'].fillna(0, inplace=True)
merged_data['Total Carbon'].fillna(0, inplace=True)

# Select the relevant columns
ground_truth = merged_data['Manufacturing2']
predictions = merged_data['Total Carbon']

# Calculate MSE, MAE, and MAPE
mse = mean_squared_error(ground_truth, predictions)
mae = mean_absolute_error(ground_truth, predictions)

# Calculate MAPE, excluding zero values from the ground truth
non_zero_mask = ground_truth != 0
mape = np.mean(np.abs((ground_truth[non_zero_mask] - predictions[non_zero_mask]) / ground_truth[non_zero_mask])) * 100

print(mse)
print(mae)
print(mape)
# print("Mean Squared Error (MSE):", mse)
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Absolute Percentage Error (MAPE):", mape, "%")
