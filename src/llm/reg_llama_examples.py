import pandas as pd
from joblib import load

# Load the saved model and preprocessor
model = load('../reg/output/reg_model.joblib')
preprocessor = load('../reg/output/preprocessor.joblib')

# Read data from the CSV file
data = pd.read_csv('output/prompt_example_specs.csv')

# Predict using the loaded model for each row in the data
predictions = model.predict(preprocessor.transform(data))

# Save the predicting result to a txt file, each value in a line
with open('output/examples_pcf_values.txt', 'w') as file:
    for pred in predictions:
        file.write(str(pred) + '\n')
