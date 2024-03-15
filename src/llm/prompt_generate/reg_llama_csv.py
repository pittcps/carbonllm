import csv
import os
import re

# Directory containing the prompt files
directory_path = "../prompts/hp_reg"  # Update this path if necessary

# Output CSV file path
csv_file_path = "../output/prompt_example_specs.csv"

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# Function to parse the input text
def parse_input(input_text):
    # Initialize default values
    product_name = "Unknown"
    processor_cores = 0
    memory = 0
    ssd = 0
    power = 0
    hdd = 0

    lines = input_text.split('. ')
    for line in lines:
        if "product is" in line:
            product_name = line.split("product is ")[1]
        elif "HDD size" in line:
            hdd = float(line.split(" HDD size is ")[1].split(" ")[0])
        elif "power supply" in line:
            power = float(line.split(" power supply is ")[1].split(" ")[0])
        elif "SSD size" in line:
            ssd = float(line.split(" SSD size is ")[1].split(" ")[0])
        elif "number of processor cores" in line:
            processor_cores = float(line.split("number of processor cores is ")[1].split(".")[0])
        elif "memory size" in line:
            memory = float(line.split("memory size is ")[1].split(" ")[0])

    return product_name, processor_cores, memory, ssd, power, hdd

# Prepare to write to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Commercial Name', 'Processor Cores', 'Memory', 'SSD', 'Power', 'HDD'])

    # Loop through files in the directory
    for filename in sorted(os.listdir(directory_path), key=extract_number):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                # Splitting the content by 'input:\n' and taking the last part
                parts = content.split('input:\n')
                last_input = parts[-1] if len(parts) > 1 else None
                if last_input:
                    input_text = last_input.strip()
                    # Parse the input text
                    product_name, processor_cores, memory, ssd, power, hdd = parse_input(input_text)
                    # Write the parsed data to CSV
                    writer.writerow([product_name, processor_cores, memory, ssd, power, hdd])

print(f"CSV file created at {csv_file_path}")
