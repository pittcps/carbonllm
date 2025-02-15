import textwrap
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import csv
import ast
import time

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

model = genai.GenerativeModel('gemini-2.0-flash')

def extract_answer(response_text):
    answer_match = re.search(r"\[(.*?)\]", response_text, re.DOTALL)
    if answer_match:
        answer_row = "[" + answer_match.group(1) + "]"
        try:
            answer_list = ast.literal_eval(answer_row)
            if isinstance(answer_list, list):
                if all(isinstance(item, (int, float)) for item in answer_list):
                    return answer_list
                elif all(isinstance(item, dict) and len(item) == 1 and isinstance(list(item.values())[0], (int, float)) for item in answer_list):
                    return [list(item.values())[0] for item in answer_list]
        except (ValueError, SyntaxError):
            pass
    return None

def get_row_count(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) <= 1:
                return 0
            return len(rows) - 1
    except FileNotFoundError:
        return 0

output_file_path = '../output/gemini1.5_RAG_output.csv'
answer_file_path = '../output/gemini1.5_RAG_answers.csv'
input_file_path = '../input/test.csv'

processed_count = get_row_count(answer_file_path)

with open(output_file_path, 'a', newline='') as output_file, \
     open(answer_file_path, 'a', newline='') as answer_file, \
     open(input_file_path, 'r') as input_file:

    output_writer = csv.DictWriter(output_file, fieldnames=['Prompt', 'Output'])
    answer_writer = csv.DictWriter(answer_file, fieldnames=['Answer'])

    if processed_count == 0:
        output_writer.writeheader()
        answer_writer.writeheader()
    elif processed_count == 1:
        with open(answer_file_path, 'r') as answer_check:
            reader = csv.reader(answer_check)
            rows = list(reader)
            if len(rows) == 1:
                processed_count = 0

    input_reader = csv.DictReader(input_file)
    for _ in range(processed_count):
        next(input_reader)

    for row in input_reader:
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

        print('PROMPT')
        print(eval_prompt)
        time.sleep(5)
        response = model.generate_content(eval_prompt)
        orig_output = response.text
        print('ORIGINAL OUTPUT')
        print(orig_output)

        answer_row = extract_answer(orig_output)
        if not answer_row:
            answer_row = []
        print('ANSWER LIST')
        print(answer_row)

        answer_writer.writerow({'Answer': answer_row})
        output_writer.writerow({'Prompt': eval_prompt, 'Output': orig_output})

        answer_file.flush()
        output_file.flush()
