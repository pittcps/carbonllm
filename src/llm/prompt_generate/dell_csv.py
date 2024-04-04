import csv
import os
import re
import json

# Directory containing the prompt files
# directory_path = "../prompts/dell"
directory_path = "../prompts/all"

# Output CSV file path
csv_file_path = "../output/prompt_example_specs.csv"

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def parse_input_to_json(input_text):
    # Initialize default structure
    result = {
        "Model": [{"quantity": "Unknown"}],
        "RAM": [{"quantity": 0, "unit": "GB"}],
        "SSD": [{"quantity": 0, "unit": "GB"}],
        "Display": [{"quantity": 0, "unit": "inch"}]
    }

    lines = input_text.split('. ')
    for line in lines:
        if "Model is" in line:
            result["Model"][0]["quantity"] = line.split("Model is ")[1]
        elif "SSD is" in line:
            result["SSD"][0]["quantity"] = float(line.split(" SSD is ")[1].split(" ")[0])
        elif "RAM is" in line:
            result["RAM"][0]["quantity"] = float(line.split(" RAM is ")[1].split(" ")[0])
        elif "Display is" in line:
            result["Display"][0]["quantity"] = float(line.split(" Display is ")[1].split(" ")[0])

    return result

# Prepare to write to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Model', 'RAM', 'SSD', 'Display'])

    # Loop through files in the directory
    for filename in sorted(os.listdir(directory_path), key=extract_number):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                parts = content.split('input:\n')
                last_input = parts[-1] if len(parts) > 1 else None
                if last_input:
                    # Parse the last input section to JSON
                    product_specs = parse_input_to_json(last_input)
                    model = product_specs["Model"][0]["quantity"]
                    ram = product_specs["RAM"][0]["quantity"]
                    ssd = product_specs["SSD"][0]["quantity"]
                    display = product_specs["Display"][0]["quantity"]
                    writer.writerow([model, ram, ssd, display])

print(f"CSV file has been created at {csv_file_path}")
