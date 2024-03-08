import pandas as pd
from dram_model import Fab_DRAM
from hdd_model import Fab_HDD
from ssd_model import Fab_SSD
from logic_model import Fab_Logic

def calculate_ACT_carbon(ssd_capacity):
    ic_yield = 1
    cpu_area = 0  # cm^2

    # Create instances of the different components
    CPU_Logic = Fab_Logic(gpa="95", carbon_intensity="src_coal", process_node=28, fab_yield=ic_yield)
    SSD_main = Fab_SSD(config="nand_30nm", fab_yield=ic_yield)
    DRAM_SSD_main = Fab_DRAM(config="ddr3_50nm", fab_yield=ic_yield)

    # Setting the area and capacities
    CPU_Logic.set_area(cpu_area)
    SSD_main.set_capacity(ssd_capacity)
    DRAM_SSD_main.set_capacity(0)  # Setting to 0 as no specific value is provided

    # Packaging footprint (assumed to be 0 for simplicity)
    packaging_intensity = 0  # gram CO2
    SSD_main_packaging = packaging_intensity

    # Compute end-to-end carbon footprints
    SSD_main_co2 = (SSD_main.get_carbon() + DRAM_SSD_main.get_carbon() + SSD_main_packaging) / 1000.
    ACT_carbon = SSD_main_co2

    return ACT_carbon

# Load the data from the CSV file
file_path = 'matched_products.csv'  # Update with the correct file path
hp_data = pd.read_csv(file_path)

# Create a new dataframe for the output
output_data = pd.DataFrame(columns=['Commercial Name', 'ACT_carbon'])

# Iterate over each row in the hp_data dataframe
for _, row in hp_data.iterrows():
    ssd_value = row['SSD'] if pd.notna(row['SSD']) else 0

    # Calculate ACT_carbon for the current row
    act_carbon = calculate_ACT_carbon(ssd_value)

    # Append the results to the output dataframe
    output_data = output_data.append({
        'Commercial Name': row['Commercial Name'],
        'ACT_carbon': act_carbon
    }, ignore_index=True)

# Save the results to a CSV file
output_file_path = 'ACT_carbon_output_2.csv'  # Update with the desired output file path
output_data.to_csv(output_file_path, index=False)
