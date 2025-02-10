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

model = genai.GenerativeModel('gemini-1.5-flash')

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
        eval_prompt = f"""You'll be provided with some questions and a reference. First, you must check whether the reference is relevant to the question. If the reference is relevant, provide the answer of list type.
Here are some examples.

Example 1:
### Question: What are the carbon footprints of chassis, total, and display in the R856T-TCO laptop?
### Reference:                  Product Carbon Footprint Acer carefully consider environmental factors in every stage of the product life cycle. This includes selecting materials during design, through packaging and shipping, to usage and recycling to reduce environmental impacts. Acer uses PAIA (Product Attribute to Impact Algorithm) to perform product carbon footprints. The PAIA platform, developed based on MIT‚Äôs methodology, was created to speed up the process while delivering streamlined and consistent results that are robust enough to make fact based decisions on product sustainability. 231 +/- 46‚Ä†kgCO2e Estimated carbon footprint Acer Chromebook Spin 512 ‚Ä†All estimates of carbon footprint are uncertain. For this product, the 5th and 95th percentile of the carbon footprint estimate, 140 kgCO2e and 430 kgCO2e, to reflect that uncertainty. That estimate has a mean of 231 kg of CO2e and standard deviation of 46 kg of CO2e . R856T, R856TN, R856LT, R856LTN R856T-TCO, R856TN-TCO, R856LT-TCO, R856LTN-TCO Product carbon footprint by percentag  e % 47.3% 17.5% 10.0% 8.5% 6.3% 5.8% 3.2% 0.8% 0.6% 0.0% 0.0% General Information 1.4 kg 12"" 11.3 kWh 4 years About the Data Disclaimer 2023/May The product carbon footprint was calculate using the Product Attribute to Impact Algorithm model, Notebook tool, version 1.3.2, copyright by the ICT Benchmarking collaboration including the Massachusetts Institute of Technology's Materials Systems Laboratory and partners. The LCA result strongly influenced by the assumptions made and PAIA tools are not configured to allow for simultaneous simulation, it is not recommended that PAIA results be used in comparisons. Product breakout Display Mainboard (and other boards) Use Power Supply Unit(s) Transport Chassis    Typical Energy Consumption (Yearly TEC) Battery Packaging End of Life    Final Assembly in China and use in Europe 0.00 0.00 Learn more about Acer Sustainability, please visit Acer Sustainability Website and Acer Earthion Website. All estimates of carbon footprint are uncertain. This information sheet contains a description of the carbon footprint data for this declared product, which is based on estimates of the current state of the product life cycle, but is subject to known or unknown risks or uncertainties, so actual results may be different from the statement. The information contained herein is subject to change without notice and Acer Inc. shall not be liable for technical or editorial errors or omissions contained herein.    Panel Size    Product Weight (excluded accessory and packaging)    Product Lifetime Manufacturing  83.1% End of Life 0.6% Use 10% Transport 6.3%
### Answer: [13.398, 231.0, 109.263]

Example 2:
### Question: What are the carbon footprints of manufacturing and mainboard in the HP 285 G6 Microtower PC ENERGY STAR desktop?
### Reference: Product Carbon Footprint Report 21-Aug-2023 HP ProOne 400 G6/440 G6/Zhan 66 Pro G3 G6 24 All-in-One PC ENERGY STAR GHG Emissions Manufacturing Breakout  Mainboard and other boards 37% Display 28% Solid State Drive (SSD) 13% Chassis 7% Others* 5% Power Supply Unit & External Cables 3% External components (Keyboard & Mouse) 2% Packaging 1% Hard Drive (HDD) 1% Optical Disk Drive (ODD) 1% Assumptions  5 North America 62.04 7.70 China Learn more at HP‚Äôs Sustainability Website Additional information about HP‚Äôs carbon footprinting program can  be found in HP‚Äôs yearly Sustainability Report, which is available on  the HP Sustainability website. The site also contains IT Eco Declarations, which provide product- specific environmental information, as well as information on HP‚Äôs  product recycling programs. * Others section includes assembly energy, other subassemblies  and all subassemblies packaging and transport Additional product environmental performance Estimated impact As part of HP‚Äôs commitment to continually improve the environmental performance of our products, we utilize product carbon footprinting (PCF) to  better understand environmental impacts that occur at different stages of the product life cycle.  A product carbon footprint is defined as the total  amount of greenhouse gases emitted directly and indirectly by a product over its lifetime. Greenhouse gas emissions are reported as global warming  potential for 100-year time horizon (GWP-100) in units of CO2 equivalence. Our product carbon footprints include full value chain emissions, which  incorporate carbon emissions due to raw materials extraction, manufacturing, distribution, use, and product end-of-use. The information provided here represents the lifecycle carbon footprint of an industry-average desktop computer  with the specifications listed in  Assumptions table. HP's environmental impact calculations are done in accordance with ISO 14040/44. All estimates of impact results are uncertain, resulting largely  from data limitations and data quality. To mitigate this uncertainty, HP has developed HP-specific tools that use a combination of HP processes and  product data, as well as high-quality lifecycle assessment data. HP strives to provide the most accurate environmental impact results but uncertainty  will never be completely minimized and results should be considered accordingly. Lifetime of product (years) Use location Use energy demand (kWh/year)  Product weight (kg) Final manufacturing location¬© Copyright 2021 HP Development Company, L.P. The information contained herein is subject to  change without notice. The only warranties for HP products and services are set forth in the express warranty statements accompanying such products and services. Nothing herein should be c ons trued as constituting an additional warranty. HP shall not be liable for technical or editorial errors or omissions contained herein. HP shall not be liable for technical or editorial errors or omissions contained herein.395 395kg CO 2 eq. eq. Manufacturing 57% Distribution 0% Use 42% End of Life 1% Value chain  carbon footprint
### Answer: []


Now the questions and reference are shown below. What are the answers to the questions?
### Question: {row['Question']}
### Reference: {row['Reference text']}
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
