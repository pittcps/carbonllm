import os
import re
from llama_cpp import Llama
from joblib import load
import pandas as pd

# Load the saved model and preprocessor
model = load('../reg/output/reg_model.joblib')
preprocessor = load('../reg/output/preprocessor.joblib')

# Define Llama model
LLM = Llama(model_path="/data/kaz81/src/llama-2-7b.Q4_K_M.gguf", n_ctx=10000, n_gpu_layers=1)

def extract_values_and_predict(output):
    """
    Extracts numerical values after specified texts from the output and uses them for prediction.
    """
    # Patterns to match each line of interest
    patterns = {
        'Processor Cores': r'"Number of processor cores": \[\{"quantity": (\d+(\.\d+)?)\}\]',
        'Memory': r'"Memory size": \[\{"quantity": (\d+(\.\d+)?), "unit": "GB"\}\]',
        'SSD': r'"SSD size": \[\{"quantity": (\d+(\.\d+)?), "unit": "GB"\}\]',
        'Power': r'"Power supply": \[\{"quantity": (\d+(\.\d+)?), "unit": "W"\}\]',
        'HDD': r'"HDD size": \[\{"quantity": (\d+(\.\d+)?), "unit": "GB"\}\]',
    }

    input_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            input_data[key] = [float(match.group(1))]
        else:
            input_data[key] = [0.0]  # Default value in case of no match

    df = pd.DataFrame.from_dict(input_data)
    transformed_input = preprocessor.transform(df)
    prediction = model.predict(transformed_input)
    return prediction[0]

def run_llama_on_file(file_path):
    output_string = ""

    with open(file_path, 'r') as file:
        prompt = file.read()
        output_string += "\nLlama Input:\n" + prompt

    output1 = LLM(prompt, max_tokens=0)
    output = output1["choices"][0]["text"]
    print("\nLlama Output:\n")
    print(output)
    output_string += "\nLlama Output:\n" + output

    result = extract_values_and_predict(output)
    output_string += f"\nPredicted PCF: {result}\n"
    print(f"\nPredicted PCF: {result}\n")

    return result, output_string

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

directory = 'prompts/hp_reg'
results = []
output = ""
for filename in sorted(os.listdir(directory), key=extract_number):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        r, output_string = run_llama_on_file(file_path)
        results.append(r)
        output += output_string

pcf_output_file_path = 'output/pcf_values_llama.txt'
with open(pcf_output_file_path, 'w') as file:
    for result in results:
        file.write(str(result) + "\n")

pcf_output_file_path = 'output/output_llama.txt'
with open(pcf_output_file_path, 'w') as file:
    file.write(output)
