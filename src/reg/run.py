from joblib import load
import pandas as pd

# Load the saved model and preprocessor
model = load('output/reg_model.joblib')
preprocessor = load('output/preprocessor.joblib')

# Example customized input
custom_input = pd.DataFrame({
    'Processor Cores': [4],  # Example values
    'Memory': [16],          # in GB
    'SSD': [256],            # in GB
    'Power': [450],          # in Watts
    'HDD': [1024]            # in GB
})

# Transform the input using the loaded preprocessor
custom_input_transformed = preprocessor.transform(custom_input)

# Predict using the loaded model
custom_pred = model.predict(custom_input_transformed)

# Print the prediction
print(f"Predicted PCF: {custom_pred[0]}")
