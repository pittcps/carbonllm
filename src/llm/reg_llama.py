import os
import re
import json
from llama_cpp import Llama
from joblib import load
import pandas as pd

# Load the saved model and preprocessor
model = load('../reg/output/reg_model.joblib')
preprocessor = load('../reg/output/preprocessor.joblib')

# Define Llama model
LLM = Llama(model_path="/data/kaz81/src/llama-2-7b.Q4_K_M.gguf", n_ctx=10000, n_gpu_layers=1)

def parse_json_and_predict(json_output):
    """
    Parses the JSON output to extract model input features,
    transforms these features, and predicts the PCF value.
    """
    data = json.loads(json_output)
    input_data = {
        'Processor Cores': [data["Number of processor cores"][0]["quantity"]],
        'Memory': [data["Memory size"][0]["quantity"]],
        'SSD': [data["SSD size"][0]["quantity"]],
        'Power': [data["Power supply"][0]["quantity"]],
        'HDD': [data["HDD size"][0]["quantity"]]
    }
    df = pd.DataFrame.from_dict(input_data)
    transformed_input = preprocessor.transform(df)
    prediction = model.predict(transformed_input)
    return prediction[0]

# Assuming the rest of your code remains mostly unchanged,
# here's where you integrate the new function:

def run_llama_on_file(file_path):
    output_string = ""
    result = 0.0

    with open(file_path, 'r') as file:
        prompt = file.read()
        output_string += "\nLlama Input:\n"
        output_string += prompt
        # print(prompt)

    output1 = LLM(prompt, max_tokens=0)
    output = output1["choices"][0]["text"]
    print("Llama Output:")
    print(output)
    output_string += "\nLlama Output:\n"
    output_string += output

    try:
        formatted_output = output.replace("output:", "").strip()
        # json_output = json.loads(formatted_output)

        # Parse the JSON output and predict
        result = parse_json_and_predict(formatted_output)  # Assuming this function expects a dictionary
        output_string += f"Predicted PCF: {result}\n"
        print(f"Predicted PCF: {result}")
    except Exception as e:
        output_string += f"An error occurred during execution: {str(e)}\n"
        print(f"An error occurred: {str(e)}")

    return result, output_string

# Extracts the numerical part of the filename and converts it to an integer
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# Directory containing prompts txt files
directory = 'prompts/hp_reg'

# Iterate through each file in the directory, sorted by numerical value in filename
results = []
output = ""
for filename in sorted(os.listdir(directory), key=extract_number):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        r, output_string = run_llama_on_file(file_path)
        results.append(r)
        output += output_string

pcf_output_file_path = f'output/pcf_values_llama.txt'
with open(pcf_output_file_path, 'w') as file:
    for r1 in results:
        file.write(str(r1) + "\n")

pcf_output_file_path = f'output/output_llama.txt'
with open(pcf_output_file_path, 'w') as file:
    file.write("\n" + output + "\n")
