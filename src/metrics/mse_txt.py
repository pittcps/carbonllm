import numpy as np
import matplotlib.pyplot as plt

# GT
with open('../llm/output/pcf_values.txt', 'r') as file:
    ground_truth = np.array([float(line.strip()) for line in file])
# Predicted

# with open('../llm/output/pcf_values_llama.txt', 'r') as file:
#     predicted_values = np.array([float(line.strip()) for line in file])
# with open('../llm/output/examples_pcf_values.txt', 'r') as file:
#     predicted_values = np.array([float(line.strip()) for line in file])
# with open('../llm/output/pcf_values_llama_combined.txt', 'r') as file:
#     predicted_values = np.array([float(line.strip()) for line in file])
with open('../llm/output/avg_examples_pcf_values.txt', 'r') as file:
    predicted_values = np.array([float(line.strip()) for line in file])
# with open('../llm/output/avg_pcf_values_llama_combined.txt', 'r') as file:
#     predicted_values = np.array([float(line.strip()) for line in file])

# Compute Mean Squared Error (MSE)
mse = np.mean((predicted_values - ground_truth) ** 2)

# Compute Mean Absolute Error (MAE)
mae = np.mean(np.abs(predicted_values - ground_truth))

# Compute Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((predicted_values - ground_truth) / ground_truth)) * 100

# Plot actual vs. predicted values with a diagonal line
plt.scatter(ground_truth, predicted_values, alpha=0.5)
plt.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted PCF')
# plt.savefig('output/llama_plot.png')
# plt.savefig('output/examples_plot.png')
# plt.savefig('output/0_combined_plot.png')
plt.savefig('output/avg_plot.png')
# plt.savefig('output/avg_combined_plot.png')
plt.close()

# Print computed metrics
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}%')
