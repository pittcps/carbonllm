import pandas as pd
import json

def create_jsonl(input_file_path, output_file_path):
    data = pd.read_csv(input_file_path)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        custom_id_counter = 1
        
        for index, row in data.iterrows():
            json_entry = {
                "custom_id": f"request-{custom_id_counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [
                        {"role": "system", "content": f"""You'll be provided with some questions and a reference. First, you must check whether the reference is relevant to the question. If the reference is relevant, provide the answer of list type. The answer is enclosed by square brackets. Only return the list of float numbers without units.
Here are some examples.

Example 1:
### Question: What are the carbon footprints of chassis, total, and display in the R856T-TCO laptop?
### Reference:                  Product Carbon Footprint Acer carefully consider environmental factors in every stage of the product life cycle. This includes selecting materials during design, through packaging and shipping, to usage and recycling to reduce environmental impacts ... 231 +/- 46‚Ä†kgCO2e Estimated carbon footprint Acer Chromebook Spin 512 ‚Ä†All estimates of carbon footprint are uncertain. For this product, the 5th and 95th percentile of the carbon footprint estimate, 140 kgCO2e and 430 kgCO2e, to reflect that uncertainty. That estimate has a mean of 231 kg of CO2e and standard deviation of 46 kg of CO2e . R856T, R856TN, R856LT, R856LTN R856T-TCO, R856TN-TCO, R856LT-TCO, R856LTN-TCO Product carbon footprint by percentag  e % 47.3% 17.5% 10.0% 8.5% 6.3% 5.8% 3.2% 0.8% 0.6% 0.0% 0.0% General Information 1.4 kg 12"" 11.3 kWh 4 years About the Data Disclaimer ...    Panel Size    Product Weight (excluded accessory and packaging)    Product Lifetime Manufacturing  83.1% End of Life 0.6% Use 10% Transport 6.3%
### Answer: [13.398, 231.0, 109.263]

Example 2:
### Question: What are the carbon footprints of manufacturing and mainboard in the HP 285 G6 Microtower PC ENERGY STAR desktop?
### Reference: Product Carbon Footprint Report 21-Aug-2023 HP ProOne 400 G6/440 G6/Zhan 66 Pro G3 G6 24 All-in-One PC ENERGY STAR GHG Emissions Manufacturing Breakout  Mainboard and other boards 37% Display 28% Solid State Drive (SSD) 13% Chassis 7% Others* 5% Power Supply Unit & External Cables 3% External components (Keyboard & Mouse) 2% Packaging 1% Hard Drive (HDD) 1% Optical Disk Drive (ODD) 1% Assumptions  5 North America 62.04 7.70 China Learn more at HP‚Äôs Sustainability Website Additional information about HP‚Äôs carbon footprinting program can  be found in HP‚ ... HP shall not be liable for technical or editorial errors or omissions contained herein.395 395kg CO 2 eq. eq. Manufacturing 57% Distribution 0% Use 42% End of Life 1% Value chain  carbon footprint
### Answer: []


"""},
                        {"role": "user", "content": f"""Now the questions and reference are shown below. What are the answers to the questions?
### Question: {row['Question']}
### Reference: {row['Reference text']}
### Answer:"""}
                    ],
                    "max_tokens": 30
                }
            }
            f.write(json.dumps(json_entry) + '\n')
            custom_id_counter += 1

input_file_path = '../output/test.csv'
output_file_path = '../output/gpt4o_test.jsonl'

create_jsonl(input_file_path, output_file_path)