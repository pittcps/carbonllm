import os
import re

from llama_cpp import Llama

# Define Llama model
LLM = Llama(model_path="/data/kaz81/src/llama-2-7b-chat.ggmlv3.q8_0.bin", n_ctx=10000, n_gpu_layers=1)

def run_llama_on_file(file_path):
    output_string = ""

    # Read the content of the file
    with open(file_path, 'r') as file:
        prompt = file.read()

    response = ""
    result = 0.0
    execution_successful = False

    while not execution_successful:
        # Run Llama model
        output = LLM(prompt, max_tokens=0)
        response = output["choices"][0]["text"]

        print(response)
        output_string += response

        match = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
        if match:
            python_code = match.group(1)
            # Define a global dictionary to execute the code in
            global_namespace = {}
            try:
                # Execute the Python code
                exec(python_code, global_namespace)
                # Check if 'Total_carbon' is in the global namespace and print it
                if 'Total_carbon' in global_namespace:
                    execution_successful = True
                    result = global_namespace['Total_carbon']
                    print('Total_carbon:', result)
                else:
                    execution_successful = True
                    print("The variable 'Total_carbon' does not exist.")
            except Exception as e:
                execution_successful = True
                # Handle the exception and continue the program
                # return f"An error occurred during execution: {str(e)}"
        else:
            execution_successful = True
            print(f'Python program not found for {file_path}.')
            output_string += f'Python program not found for {file_path}.'

    if execution_successful:
        print("Result is:", result)
        output_string += f"Result is:{result}"
    else:
        print("Execution was not successful.")
        output_string += "Execution was not successful."
    return response, result, output_string

# Extracts the numerical part of the filename and converts it to an integer
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# Directory containing prompts txt files
directory = 'out/prompts'

# Iterate through each file in the directory, sorted by numerical value in filename
results = []
resps = []
output = ""
for filename in sorted(os.listdir(directory), key=extract_number):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        resp, r, output_string = run_llama_on_file(file_path)
        results.append(r)
        resps.append(resp)
        output += output_string

pcf_output_file_path = f'out/pcf_m_values_llama_1.txt'
with open(pcf_output_file_path, 'w') as file:
    for r1 in results:
        file.write(str(r1) + "\n")

pcf_output_file_path = f'out/output_llama_1.txt'
with open(pcf_output_file_path, 'w') as file:
    file.write("\n" + output + "\n")
