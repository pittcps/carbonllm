import pandas as pd
import os
import math

def generate_and_save_random_prompts(csv_file_path, num_samples, num_examples_per_sample):
    df = pd.read_csv(csv_file_path)
    formatted_prompts = []

    for i in range(num_samples):
        selected_rows = df.sample(n=num_examples_per_sample)
        additional_row = df.sample(n=1)
        prompts = []

        start_string = f"You will compute product carbon footprint (kg CO2e). Return the analysis and the final answer.\n"
        start_string += "The provided carbon-specs information is as follows:\nThe average mainboard carbon footprint for processor core number 2 is 64.543 kg CO2e. The one for 4 is 63.381 kg CO2e.\nThe average SSD carbon footprint for SSD storage 128 GB is 39.210 kg CO2e. The one for 240 GB is 57.077 kg CO2e. The one for 256 GB is 45.246 kg CO2e.\nThe average power supply unit carbon footprint for power supply 45 W is 3.708 kg CO2e. The one for 65 W is 4.473 kg CO2e. The one for 90 W is 4.663 kg CO2e.\nThe average battery carbon footprint for power supply 45 W is 4.662 kg CO2e. The one for 65 W is 4.946 kg CO2e.\nThe display carbon footprint is equal to 0.05*(screen size)^2 + 1.67*(screen size) - 3.59 kg CO2e if screen size is not 0\". If screen size is 0\", the display carbon footprint is 0 kg CO2e.\nThe others carbon footprint on average is 53.503 kg CO2e.\nThe packaging carbon footprint is equal to ((Total carbon) - 113.45) / 49.75.\n\n"
        start_string += f"Here are {num_examples_per_sample} examples on how to do it."
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

            # Processor core comparison
            prompt = f"Q:\nThe product name is {product_name}\nThe processor core number is {cores_num}\nThe SSD storage is {ssd_storage} GB\nThe power supply is {power_supply} W\nThe screen size is {screen_size}\"\n\n"
            prompt += "A:\n"

            core_footprint = 0
            if cores_num > 0:
                mainboard_diff_2 = abs(2 - cores_num)
                mainboard_diff_4 = abs(4 - cores_num)
                selected_core = 4 if mainboard_diff_4 < mainboard_diff_2 else 2
                core_footprint = 63.381 if selected_core == 4 else 64.543
                prompt += f"The processor core number is {cores_num}. |2 - {cores_num}| = {mainboard_diff_2}. |4 - {cores_num}| = {mainboard_diff_4}. The smallest number is {min(mainboard_diff_2, mainboard_diff_4)}, which processor core number is {selected_core}. Therefore, we use the average mainboard carbon footprint for {selected_core} to estimate the mainboard carbon footprint. It is {core_footprint} kg CO2e.\n"
            else:
                prompt += "The processor core number is 0. Therefore, the mainboard carbon footprint is 0 kg CO2e.\n"

            # SSD storage analysis
            ssd_footprint = 0
            if ssd_storage > 0:
                ssd_diff_128 = abs(128 - ssd_storage)
                ssd_diff_240 = abs(240 - ssd_storage)
                ssd_diff_256 = abs(256 - ssd_storage)
                ssd_sizes = [(128, ssd_diff_128, 39.210), (240, ssd_diff_240, 57.077), (256, ssd_diff_256, 45.246)]
                closest_ssd, _, ssd_footprint = min(ssd_sizes, key=lambda x: x[1])
                prompt += f"The SSD storage is {ssd_storage} GB. |128 - {ssd_storage}| = {ssd_diff_128}. |240 - {ssd_storage}| = {ssd_diff_240}. |256 - {ssd_storage}| = {ssd_diff_256}. The smallest number is {min(ssd_diff_128, ssd_diff_240, ssd_diff_256)}, which SSD storage is {closest_ssd}. Therefore, we use the average SSD carbon footprint for {closest_ssd} to estimate the mainboard carbon footprint. It is {ssd_footprint} kg CO2e.\n"
            else:
                prompt += "The SSD storage is 0 GB. Therefore, the SSD carbon footprint is 0 kg CO2e.\n"

            # Power supply and battery footprint analysis
            power_footprint = 0
            battery_footprint = 0
            if power_supply > 0:
                power_diff_45 = abs(45 - power_supply)
                power_diff_65 = abs(65 - power_supply)
                power_diff_90 = abs(90 - power_supply)
                power_sizes = [(45, power_diff_45, 3.708), (65, power_diff_65, 4.473), (90, power_diff_90, 4.663)]
                closest_power, _, power_footprint = min(power_sizes, key=lambda x: x[1])
                prompt += f"The power supply is {power_supply} W. |45 - {power_supply}| = {power_diff_45}. |65 - {power_supply}| = {power_diff_65}. |90 - {power_supply}| = {power_diff_90}. The smallest number is {min(power_diff_45, power_diff_65, power_diff_90)}, which power supply is {closest_power}. Therefore, we use the average power supply unit carbon footprint for {closest_power} to estimate the power supply unit carbon footprint. It is {power_footprint} kg CO2e.\n"

                battery_diff_45 = abs(45 - power_supply)
                battery_diff_65 = abs(65 - power_supply)
                battery_sizes = [(45, battery_diff_45, 4.662), (65, battery_diff_65, 4.946)]
                closest_battery, _, battery_footprint = min(battery_sizes, key=lambda x: x[1])
                prompt += f"The power supply is {power_supply} W. |45 - {power_supply}| = {power_diff_45}. |65 - {power_supply}| = {power_diff_65}. The smallest number is {min(battery_diff_45, battery_diff_65)}, which power supply is {closest_battery}. Therefore, we use the average battery carbon footprint for {closest_battery} to estimate the battery carbon footprint. It is {battery_footprint} kg CO2e.\n"
            else:
                prompt += "The power supply is 0 W. Therefore, the power supply unit and battery carbon footprints are 0 kg CO2e.\n"

            # Display carbon footprint
            display_carbon = 0
            if screen_size > 0:
                display_carbon = 0.05*(screen_size)**2 + 1.67*(screen_size) - 3.59
                display_carbon = round(display_carbon, 3)
                prompt += f"The screen size is {screen_size}\". Therefore, the display carbon footprint is equal to 0.05*({screen_size})^2 + 1.67*({screen_size}) - 3.59 = {display_carbon} kg CO2e.\n"
            else:
                prompt += "The screen size is 0\". Therefore, the display carbon footprint is 0 kg CO2e.\n"

            prompt += "The others carbon footprint is 53.503 kg CO2e.\n"

            # Total carbon footprint calculation
            total_carbon = sum([core_footprint if cores_num > 0 else 0, ssd_footprint if ssd_storage > 0 else 0, power_footprint if power_supply > 0 else 0, battery_footprint if power_supply > 0 else 0, display_carbon if screen_size > 0 else 0, 53.503])
            total_carbon = round(total_carbon, 3)
            packaging_carbon = (total_carbon - 113.45) / 49.75
            packaging_carbon = round(packaging_carbon, 3)
            final_carbon = total_carbon + packaging_carbon
            final_carbon = round(final_carbon, 3)
            prompt += f"Summing up the results above, we'll get the current total carbon footprint: {total_carbon} kg CO2e.\n"
            prompt += f"The packaging carbon footprint is equal to ({total_carbon} - 113.45) / 49.75 = {packaging_carbon} kg CO2e.\n"
            prompt += f"The answer is {total_carbon} + {packaging_carbon} = {final_carbon} kg CO2e."

            prompts.append(prompt)

        # Add the additional example without the A: part
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

            prompt = f"Q:\nThe product name is {product_name}\nThe processor core number is {cores_num}\nThe SSD storage is {ssd_storage} GB\nThe power supply is {power_supply} W\nThe screen size is {screen_size}\"\n\nA:\n?"

            prompts.append(prompt)

        # Save to a text file for each sample
        output_file_path = f'out/cot_prompts/prompt_sample_{i+1}.txt'
        directory = os.path.dirname(output_file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file_path, 'w') as file:
            for prompt in prompts:
                file.write(prompt + "\n\n")

    return [f'prompt_sample_{i+1}.txt' for i in range(num_samples)]

# Example usage
generate_and_save_random_prompts('input/hp_combined_1.csv', 3, 30)
