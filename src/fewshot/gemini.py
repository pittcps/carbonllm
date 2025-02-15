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

output_file_path = '../output/gemini1.5_output.csv'
answer_file_path = '../output/gemini1.5_answers.csv'
input_file_path = '../input/test_relevant.csv'

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
        eval_prompt = f"""You'll be provided with some questions. Provide the answer of list type.
Here are some examples.

Example 1:
### Question: What are the carbon footprints of chassis, total, and display in the R856T-TCO laptop?
### Answer: [13.398, 231.0, 109.263]

Example 2:
### Question: What are the components with the highest and lowest carbon footprint percentages in the manufacturing breakdown of the Latitude 5310 2-in-1 laptop?
### Answer: [{{'display': 37.2}}, {{'packaging': 0.4}}]

Example 3:
### Question: What are the top 5 components with the highest carbon footprint percentages in the manufacturing breakdown of the HP ZHAN 66 Pro A G4 All-in-One PC desktop?
### Answer: [{{'display': 26.8}}, {{'mainboard': 25.7}}, {{'ssd': 24.9}}, {{'chassis': 13.3}}, {{'power': 3.3}}]

Example 4:
### Question: What is the carbon footprint of total in the Lenovo L28u-30?
### Answer: [455.0]


Now the questions and reference are shown below. What are the answers to the questions?
### Question: {row['Question']}
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
