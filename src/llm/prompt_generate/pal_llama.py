import pandas as pd
import os
import math

def generate_and_save_random_prompts(csv_file_path, num_samples, num_examples_per_sample):
    df = pd.read_csv(csv_file_path)
    formatted_prompts = []
    pcf_values = []

    for i in range(num_samples):
        selected_rows = df.sample(n=num_examples_per_sample)
        additional_row = df.sample(n=1)
        prompts = []

        start_string = "You will write and return only the python program to solve carbon emission problem.\nHere are 3 examples on how to do it,\n"
        prompts.append(start_string)

        for index, row in selected_rows.iterrows():
            product_name = row['Commercial Name']
            cores_num = row.get('Processor Cores', 0)
            if math.isnan(cores_num):
                cores_num = 0
            ssd_storage = row.get('SSD', 0)
            if math.isnan(ssd_storage):
                ssd_storage = 0
            power_supply = row.get('Power', 0)
            if math.isnan(power_supply):
                power_supply = 0
            screen_size = row.get('Screen size', 0)
            if math.isnan(screen_size):
                screen_size = 0

            prompt = f"Input:\nProduct name: {product_name}\nProduct core number: {cores_num}\nSSD storage: {ssd_storage} GB\nPower supply: {power_supply} W\nScreen size: {screen_size}\"\n\n"
            prompt += "# solution in Python:\n\n"
            prompt += "```\nimport pandas as pd\n"
            prompt += f"cores = {cores_num}\nssd = {ssd_storage}\npower = {power_supply}\nscreen_size = {screen_size}\n"
            prompt += "def closest_match(key1, input_dict):\n    closest_key = min(input_dict.keys(), key=lambda k: abs(k - key1))\n    return input_dict[closest_key]\n\n"
            prompt += "cores_mainboard_dict = {2: 64.543, 4: 63.381}\n"
            prompt += "if cores == 0:\n    carbon_mainboard = 0\nelse:\n    carbon_mainboard = closest_match(cores, cores_mainboard_dict)\n"
            prompt += "ssd_ssd_dict = {128: 39.210, 240: 57.077, 256: 45.246}\n"
            prompt += "if ssd == 0:\n    carbon_ssd = 0\nelse:\n    carbon_ssd = closest_match(ssd, ssd_ssd_dict)\n"
            prompt += "power_power_dict = {45: 3.708, 65: 4.473, 90: 4.663}\n"
            prompt += "if power == 0:\n    carbon_power = 0\nelse:\n    carbon_power = closest_match(power, power_power_dict)\n"
            prompt += "power_battery_dict = {45: 4.662, 65: 4.946}\n"
            prompt += "if power == 0:\n    carbon_battery = 0\nelse:\n    carbon_battery = closest_match(power, power_battery_dict)\n"
            prompt += "if screen_size == 0:\n    carbon_display = 0\nelse:\n    carbon_display = 0.05*(screen_size)**2 + 1.67*screen_size - 3.59\n"
            prompt += "carbon_others = 53.503\n"
            prompt += "Total_carbon = carbon_mainboard + carbon_ssd + carbon_power + carbon_battery + carbon_display + carbon_others\n"
            prompt += "carbon_pack = (Total_carbon - 113.45) / 49.75\n"
            prompt += "Total_carbon += carbon_pack\n```\n"

            prompts.append(prompt)

        # Add the additional example without the Python solution
        for index, row in additional_row.iterrows():
            product_name = row['Commercial Name']
            cores_num = row.get('Processor Cores', 0)
            if math.isnan(cores_num):
                cores_num = 0
            ssd_storage = row.get('SSD', 0)
            if math.isnan(ssd_storage):
                ssd_storage = 0
            power_supply = row.get('Power', 0)
            if math.isnan(power_supply):
                power_supply = 0
            screen_size = row.get('Screen size', 0)
            if math.isnan(screen_size):
                screen_size = 0

            prompt = "\nHow about this input?\n"
            prompt += f"Input:\nProduct name: {product_name}\nProduct core number: {cores_num}\nSSD storage: {ssd_storage} GB\nPower supply: {power_supply} W\nScreen size: {screen_size}\""

            prompts.append(prompt)
            pcf_values.append(row['Manufacturing2'])

        # Save to a text file for each sample
        output_file_path = f'out/prompts/prompt_sample_{i+1}.txt'
        directory = os.path.dirname(output_file_path)

        # Check if the directory exists, create it if not
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file_path, 'w') as file:
            for prompt in prompts:
                file.write(prompt + "\n")
        pcf_output_file_path = f'out/pcf_m_values_1.txt'
        with open(pcf_output_file_path, 'w') as file:
            for pcf in pcf_values:
                file.write(str(pcf) + "\n")

    return [f'prompt_sample_{i+1}.txt' for i in range(num_samples)]

generate_and_save_random_prompts('input/hp_combined_1.csv', 30, 3)
