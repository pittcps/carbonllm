import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from unittest.mock import patch
import csv
import json
import sys

base_model_id = "meta-llama/Meta-Llama-3-8B"
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

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "../models/no-RAG")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

ft_model.eval()

answers = []
count = 0

txt_file = "../output/no_RAG_output.txt"
f = open(txt_file, "w")
original_stdout = sys.stdout
sys.stdout = f

with open('../output/no_RAG_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
with open('../output/no_RAG_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

with open('../input/test_relevant.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        orig_output = []
        answer_row = ""

        eval_prompt = f"""You'll be provided with some questions. Provide the answer of list type.
### Question: {row['Question']}
### Answer:"""
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        run_times = 0
        while True:
            with torch.no_grad():
                llm_output = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=500)[0], skip_special_tokens=True)
            orig_output.append(llm_output)

            pattern = r'### Answer:\s*(\[.*?\])'
            lst_match = re.search(pattern, llm_output, re.DOTALL)
            if lst_match:
                lst_str = lst_match.group(1).strip()
                answer_row = lst_str
                break

            run_times += 1
            if run_times == 5:
                print("Tried 5 times...")
                answer_row = "N/A"
                relevance_token = "N/A"
                count += 1
                print("Answer Count:", count)
                print("Answer is NULL")
                break

        with open('../output/no_RAG_results.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Answer'])
            writer.writerow({'Answer': answer_row})

        with open('../output/no_RAG_output.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Output'])
            writer.writerow({'Output': orig_output})

f.close()
