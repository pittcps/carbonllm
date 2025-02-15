import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from unittest.mock import patch
import csv
import sys

base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

base_model.eval()

answers = []
count = 0

txt_file = "../output/ds_output.txt"
f = open(txt_file, "w")
original_stdout = sys.stdout
sys.stdout = f

with open('../output/ds_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
with open('../output/ds_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

with open('../input/test.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        orig_output = []
        answer_row = ""

        eval_prompt = f"""You'll be provided with some questions and a reference. Based on the reference, provide the answer of list type.
Here are some examples.

Example 1:
### Question: What are the carbon footprints of chassis, total, and display in the R856T-TCO laptop?
### Reference:                  Product Carbon Footprint Acer carefully consider environmental factors in every stage of the product life cycle ... That estimate has a mean of 231 kg of CO2e and standard deviation of 46 kg of CO2e . R856T, R856TN, R856LT, R856LTN R856T-TCO, R856TN-TCO, R856LT-TCO, R856LTN-TCO Product carbon footprint by percentag  e % 47.3% 17.5% 10.0% 8.5% 6.3% 5.8% 3.2% 0.8% 0.6% 0.0% 0.0% General Information 1.4 kg 12"" 11.3 kWh 4 years About the Data Disclaimer 2023/May ... The LCA result strongly influenced by the assumptions made and PAIA tools are not configured to allow for simultaneous simulation, it is not recommended that PAIA results be used in comparisons. Product breakout Display Mainboard (and other boards) Use Power Supply Unit(s) Transport Chassis    Typical Energy Consumption (Yearly TEC) Battery Packaging End of Life    Final Assembly in China and use in Europe 0.00 0.00 ... Panel Size    Product Weight (excluded accessory and packaging)    Product Lifetime Manufacturing  83.1% End of Life 0.6% Use 10% Transport 6.3%
### Answer: [13.398, 231.0, 109.263]

Example 2:
### Question: What are the components with the highest and lowest carbon footprint percentages in the manufacturing breakdown of the Latitude 5310 2-in-1 laptop?
### Reference:      Dell Latitude 5310 2-IN-1  ... This includes the  contributions  from  materials,  manufacturing, distribution, use  and end-of-life management.    This product‚Äôs estimated carbon footprint:  299 kgCO2e +/- 58 kgCO2e  Estimated impact by lifecycle stage with breakout for manufacturing  by component:  ...  Product Weight  1.351 kg  Screen Size  13.3‚Äù  Assembly  Location  China  Product Lifetime  4 years  Use Location  EU  Energy Demand  (Yearly TEC)  19.43 kWh      Disclaimer: This PCF was calculated using the PAIA model, version 1.2.6, 2020. Results shown here are subject to change as  the tool is updated.   Manufacturing 83.1% Chassis & Assembly 4.1% Hard Drive 0.0% SSD 2.7% Power Supply 8.8% Battery 1.9% Mainboard and Other Boards 28.0% Display 37.2% Packaging 0.4%
### Answer: [{{'display': 37.2}}, {{'packaging': 0.4}}]

Example 3:
### Question: What are the top 5 components with the highest carbon footprint percentages in the manufacturing breakdown of the HP ZHAN 66 Pro A G4 All-in-One PC desktop?
### Reference: Product Carbon Footprint Report HP ZHAN 66 Pro A G4 All-in-One PC GHG Emissions Manufacturing Breakout  Display 26.8% Mainboard and other boards 25.7% Solid State Drive (SSD) 24.9% Chassis 13.3% Power Supply Unit & External Cables 3.3% Others* 3.2% External components (Keyboard & Mouse) 1.8% Packaging 1.0%   Assumptions  5 North America 72.16 7.3 23.8"" China Learn more at  ... HP shall not be liable for technical or editorial errors or omissions contained herein. HP shall not be liable for technical or editorial errors or omissions contained herein.426kg CO 2eq. kg CO2 Manufacturing54%Distribution M% Use 45% End of Life 1% Value chain  carbon footprint  1% Value chain  carbon footprint
### Answer: [{{'display': 26.8}}, {{'mainboard': 25.7}}, {{'ssd': 24.9}}, {{'chassis': 13.3}}, {{'power': 3.3}}]

Example 4:
### Question: What is the carbon footprint of total in the Lenovo L28u-30?
### Reference: Lenovo Product Carbon Footprint (PCF) Information Sheet  PC/Notebook/Monitor/Tablet  Commercial Name  Lenovo L28u-30  Model Number  65FA  Issue Date  2019-08-09  - Revised 8/15/2022 Product Environmental Attributes  (a) Product Carbon Footprint Value: 455 kg of CO2e (see Note 1 below)  (b) Product Picture: (c) Life Cycle Detail by Component & Life Stage (Pie Chart):  Note 1:   All estimates of carbon footprint are uncertain. Lenovo reports the 95th percentile of the carbon footprint  estimate to reflect that uncertainty ...
### Answer: [455.0]


Now the questions and reference are shown below. What are the answers to the questions?
### Question: {row['Question']}
### Reference: {row['Text']}
### Answer:"""
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        run_times = 0
        while True:
            with torch.no_grad():
                llm_output = tokenizer.decode(base_model.generate(**model_input, max_new_tokens=500)[0], skip_special_tokens=True)
            orig_output.append(llm_output)

            token_list = llm_output.split('### Answer:')
            if len(token_list) >= 4:
                token_str = token_list[3].strip()
                pattern = r'(\[.*?\])'
                lst_match = re.search(pattern, token_str, re.DOTALL)
                if lst_match:
                    lst_str = lst_match.group(1).strip()
                    answer_row = lst_str
                    break

            run_times += 1
            if run_times == 5:
                print("Tried 5 times...")
                answer_row = "N/A"
                count += 1
                print("Answer Count:", count)
                print("Answer is NULL")
                break

        with open('../output/ds_results.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Answer'])
            writer.writerow({'Answer': answer_row})

        with open('../output/ds_output.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Output'])
            writer.writerow({'Output': orig_output})

f.close()
