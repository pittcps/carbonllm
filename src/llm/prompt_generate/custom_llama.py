import pandas as pd
import os
import math
import random

num_components = 2

def generate_and_save_random_prompts(csv_file_path, num_samples, num_examples_per_sample, column_names, random_seed=None, random_seeds_file=None):
    # Read random seeds from the file if provided
    random_seeds = []
    if random_seeds_file:
        with open(random_seeds_file, 'r') as f:
            for line in f:
                seeds = line.strip().split()  # Split each line into seeds
                random_seeds.extend(map(int, seeds))

    df = pd.read_csv(csv_file_path)
    pcf_values = []

    # Remove outliers
    df = df[df['PCF'] <= 900]

    initial_output_format = {
        col_name: [{"quantity": 0, "unit": unit}] if unit else [{"quantity": 0}]
        for col_name, unit in column_names.items()
    }

    for i in range(num_samples):
        if random_seeds and len(random_seeds) >= 2:
            seed1 = random_seeds[i * 2]
            seed2 = random_seeds[i * 2 + 1]
        else:
            seed1 = None
            seed2 = None

        selected_rows = df.sample(n=num_examples_per_sample, random_state=seed1)
        additional_row = df.sample(n=1, random_state=seed2)

        prompts = []

        # Dynamically create the start_string including output_format based on column_names
        start_string = ("You are now an expert on computing devices. You'll have a product specs input in natural language. "
                        "Convert it into JSON format. Complete it based on your knowledge. Please note the following requirements:\n"
                        "1. The entity types included in JSON must belong to: {}.\n"
                        "2. The JSON format conforms to the following form: {},\n"
                        "3. The units used in the results are {}\n\n"
                        "Here are some examples:\n").format(list(column_names.keys()), initial_output_format, [v for v in column_names.values() if v is not None])
        prompts.append(start_string.replace("'", "\""))

        for index, row in selected_rows.iterrows():
            sentences = []
            output_format = {}
            for col_name, unit in column_names.items():
                raw_value = row.get(col_name, 0)
                # Convert to float if possible, else use raw value
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value  # Use string directly if conversion fails

                # For numeric values, check for NaN and set to 0 if True
                if isinstance(value, float) and math.isnan(value):
                    value = 0

                if unit:
                    sentences.append(f"The {col_name} is {value} {unit}.")
                    output_format[col_name] = [{"quantity": value, "unit": unit}]
                else:
                    sentences.append(f"The {col_name} is {value}.")
                    output_format[col_name] = [{"quantity": value}]

            # Shuffle the list of sentences
            random.shuffle(sentences)

            # model_value = row.get('Model', 'Unknown Model')
            # product_name_sentence = f"The Model is {model_value}."

            product_name_sentence = sentences.pop(sentences.index(next(sentence for sentence in sentences if sentence.startswith("The Model is"))))
            required_sentences = [product_name_sentence] + sentences[:min(len(sentences), num_components)]

            # Construct the prompt with selected sentences
            prompt = "input:\n" + " ".join(required_sentences) + "\noutput:\n" + str(output_format).replace("'", "\"") + "\n"
            prompts.append(prompt)

        # Add the additional example without the output
        for index, row in additional_row.iterrows():
            sentences = []
            output_format = {}
            for col_name, unit in column_names.items():
                raw_value = row.get(col_name, 0)
                # Convert to float if possible, else use raw value
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value  # Use string directly if conversion fails

                # For numeric values, check for NaN and set to 0 if True
                if isinstance(value, float) and math.isnan(value):
                    value = 0

                if unit:
                    sentences.append(f"The {col_name} is {value} {unit}.")
                    output_format[col_name] = [{"quantity": value, "unit": unit}]
                else:
                    sentences.append(f"The {col_name} is {value}.")
                    output_format[col_name] = [{"quantity": value}]

            # Shuffle the list of sentences
            random.shuffle(sentences)

            # model_value = row.get('Model', 'Unknown Model')
            # product_name_sentence = f"The Model is {model_value}."
            product_name_sentence = sentences.pop(sentences.index(next(sentence for sentence in sentences if sentence.startswith("The Model is"))))
            required_sentences = [product_name_sentence] + sentences[:min(len(sentences), num_components)]

            prompt = "\nWhat is the output for this input?\n"
            prompt += "input:\n" + " ".join(required_sentences)

            prompts.append(prompt)
            pcf_values.append(row['PCF'])

        # Save to a text file for each sample
        output_file_path = f'../prompts/all/prompt_sample_{i+1}.txt'

        directory = os.path.dirname(output_file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file_path, 'w') as file:
            for prompt in prompts:
                file.write(prompt + "\n")

        # Optionally, save PCF values to a file
        pcf_output_file_path = f'../output/pcf_values.txt'
        with open(pcf_output_file_path, 'w') as file:
            for pcf in pcf_values:
                file.write(str(pcf) + "\n")

    return [f'prompt_sample_{i+1}.txt' for i in range(num_samples)]

# column_names = {'Commercial Name': None, 'Processor Cores': None, 'Memory': 'GB', 'SSD': 'GB', 'Power': 'W', 'HDD': 'GB'}
column_names = {'Model': None, 'RAM': 'GB', 'SSD': 'GB', 'Display': 'inch'} # DELL 6 2
# All: 57 3
# column_names = {'Model': None, 'RAM': 'GB', 'SSD': 'GB'} # ACER 12 1
# column_names = {'Commercial Name': None, 'SSD': 'GB'}
# generate_and_save_random_prompts('../../input/acer_laptop.csv', 6, 3, column_names, random_seed=42, random_seeds_file="../output/random_seeds.txt")
generate_and_save_random_prompts('../../input/carbon_specs_all.csv', 57, 3, column_names, random_seed=42, random_seeds_file="../output/100_random_seeds.txt")
