import pandas as pd
import os
import math
import random

def generate_and_save_random_prompts(csv_file_path, num_samples, num_examples_per_sample, random_seed=None):
    # Set the random seed
    if random_seed is not None:
        random.seed(random_seed)

    df = pd.read_csv(csv_file_path)
    formatted_prompts = []
    pcf_values = []

    for i in range(num_samples):
        selected_rows = df.sample(n=num_examples_per_sample)
        additional_row = df.sample(n=1)
        prompts = []

        start_string = "You are now an expert on computing devices. You'll have a product specs input in natural language.  Convert it into JSON format. Complete it based on your knowledge. Please note the following requirements:\n1. The entity types included in JSON must belong to: [Number of processor cores, Memory size, SSD size, Power supply, HDD size].\n2. The JSON format conforms to the following form: {output_format},\n3. The units used in the results are [GB, W]\n\noutput_format = {\"Number of processor cores\": [{{\"quantity\": 0}}],\n\"Memory size\": [{{\"quantity\": 0, \"unit\": \"GB\"}}],\n\"SSD size\": [{{\"quantity\": 0, \"unit\": \"GB\"}}],\n\"Power supply\": [{{\"quantity\": 0, \"unit\": \"W\"}}],\n\"HDD size\": [{{\"quantity\": 0, \"unit\": \"GB\"}}]}}\n\nHere are some examples:\n"
        prompts.append(start_string)

        for index, row in selected_rows.iterrows():
            product_name = row['Commercial Name']
            cores_num = row.get('Processor Cores', 0)
            if math.isnan(cores_num):
                cores_num = 0
            mem_size = row.get('Memory', 0)
            if math.isnan(mem_size):
                mem_size = 0
            ssd_size = row.get('SSD', 0)
            if math.isnan(ssd_size):
                ssd_size = 0
            power_supply = row.get('Power', 0)
            if math.isnan(power_supply):
                power_supply = 0
            hdd_size = row.get('HDD', 0)
            if math.isnan(hdd_size):
                hdd_size = 0

            # Create a list of sentences to shuffle
            sentences = [
                f"The product is {product_name}.",
                f"The number of processor cores is {cores_num}.",
                f"The memory size is {mem_size} GB.",
                f"The SSD size is {ssd_size} GB.",
                f"The power supply is {power_supply} W.",
                f"The HDD size is {hdd_size} GB."
            ]

            # Shuffle the list of sentences
            random.shuffle(sentences)

            # Ensure that "The product is {product_name}." is always included
            required_sentences = [sentences.pop(sentences.index(f"The product is {product_name}."))]

            # Randomly select three other sentences
            for _ in range(min(len(sentences), 3)):
                required_sentences.append(sentences.pop())

            # Construct the prompt with selected sentences
            prompt = "input:\n"
            for sentence in required_sentences:
                prompt += sentence + " "

            prompt += f"\noutput:\n{{\"Number of processor cores\": [{{\"quantity\": {cores_num}}}],\n\"Memory size\": [{{\"quantity\": {mem_size}, \"unit\": \"GB\"}}],\n\"SSD size\": [{{\"quantity\": {ssd_size}, \"unit\": \"GB\"}}],\n\"Power supply\": [{{\"quantity\": {power_supply}, \"unit\": \"W\"}}],\n\"HDD size\": [{{\"quantity\": {hdd_size}, \"unit\": \"GB\"}}]}}\n"

            prompts.append(prompt)

        # Add the additional example without the output
        for index, row in additional_row.iterrows():
            product_name = row['Commercial Name']
            cores_num = row.get('Processor Cores', 0)
            if math.isnan(cores_num):
                cores_num = 0
            mem_size = row.get('Memory', 0)
            if math.isnan(mem_size):
                mem_size = 0
            ssd_size = row.get('SSD', 0)
            if math.isnan(ssd_size):
                ssd_size = 0
            power_supply = row.get('Power', 0)
            if math.isnan(power_supply):
                power_supply = 0
            hdd_size = row.get('HDD', 0)
            if math.isnan(hdd_size):
                hdd_size = 0

            # Create a list of sentences to shuffle
            sentences = [
                f"The product is {product_name}.",
                f"The number of processor cores is {cores_num}.",
                f"The memory size is {mem_size} GB.",
                f"The SSD size is {ssd_size} GB.",
                f"The power supply is {power_supply} W.",
                f"The HDD size is {hdd_size} GB."
            ]

            # Shuffle the list of sentences
            random.shuffle(sentences)

            # Ensure that "The product is {product_name}." is always included
            required_sentences = [sentences.pop(sentences.index(f"The product is {product_name}."))]

            # Randomly select three other sentences
            for _ in range(min(len(sentences), 3)):
                required_sentences.append(sentences.pop())

            prompt = "\nWhat is the output for this input?\n"

            # Construct the prompt with selected sentences
            prompt += "input:\n"
            for sentence in required_sentences:
                prompt += sentence + " "

            prompts.append(prompt)
            pcf_values.append(row['PCF'])

        # Save to a text file for each sample
        output_file_path = f'../prompts/hp_reg/prompt_sample_{i+1}.txt'
        directory = os.path.dirname(output_file_path)

        # Check if the directory exists, create it if not
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file_path, 'w') as file:
            for prompt in prompts:
                file.write(prompt + "\n")
        pcf_output_file_path = f'../output/pcf_values.txt'
        with open(pcf_output_file_path, 'w') as file:
            for pcf in pcf_values:
                file.write(str(pcf) + "\n")

    return [f'prompt_sample_{i+1}.txt' for i in range(num_samples)]

generate_and_save_random_prompts('../../input/hp_combined_hdd.csv', 30, 3, random_seed=42)
