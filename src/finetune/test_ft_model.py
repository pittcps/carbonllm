import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "NousResearch/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "llama3-8b-llama-finetune/checkpoint-4200")

eval_prompt = """You'll be provided with a question and its reference. Find out the necessary evidence from the reference to answer the question. Only return the evidence and its index in the reference.
### Question: HP ProBook 445 14 inch G9 Notebook PC: It has 1.7 kg product weight, 4.0 processor cores, 65.0 power supply, 32.0 GB RAM, 15.6 inch display, 128.0 GB SSD. What is its manufacturing carbon footprint?
### Reference: Product Carbon Footprint Report HP ProBook 445 14 inch G9 Notebook PC GHG Emissions Manufacturing Breakout  Solid State Drive (SSD) 40.0% Display 22.6% Mainboard and other boards 21.1% Chassis 7.9% Batteries 3.2% Power Supply Unit & External Cables 2.7% Others* 2.2% Packaging 0.4%   Assumptions  4 North America 13.93 1.7 15.6" China Learn more at HP‚Äôs Sustainability Website Additional information about HP‚Äôs carbon footprinting program can  be found in HP‚Äôs yearly Sustainability Report, which is available on  the HP Sustainability website. The site also contains IT Eco Declarations, which provide product- specific environmental information, as well as information on HP‚Äôs  product recycling programs. * Others section includes assembly energy, other subassemblies  and all subassemblies packaging and transport Additional product environmental performance Estimated impact As part of HP‚Äôs commitment to continually improve the environmental performance of our products, we utilize product carbon footprinting (PCF) to better  understand environmental impacts that occur at different stages of the product life cycle. A product carbon footprint is defined as the total amount of  greenhouse gases emitted directly and indirectly by a product over its lifetime. Our product carbon footprints include full value chain emissions, which  incorporates carbon emissions due to raw materials extraction, manufacturing, distribution, use, and product end-of-use. The information provided here represents the lifecycle carbon footprint of an industry-average notebook computer  with the specifications listed in  Assumptions table. HP's environmental impact calculations are done in accordance with ISO 14040/44. All estimates of impact results are uncertain, resulting largely from  data limitations and data quality. To mitigate this uncertainty, HP has developed HP-specific tools that use a combination of HP processes and product  data, as well as high-quality lifecycle assessment data. HP strives to provide the most accurate environmental impact results but uncertainty will never  be completely minimized and results should be considered accordingly. Lifetime of product (years) Use location Use energy demand (kWh/year)  Product weight (kg) Screen size (in) Final manufacturing location¬© Copyright 2021 HP Development Company, L.P. The information contained herein is subject to  change without notice. The only warranties for HP products and services are set forth in the express warranty statements accompanying such products and services. Nothing herein should be construed as constituting an ad dit ional warranty.HP shall not be liable for technical or editorial errors or omissions contained herein. HP shall not be liable for technical or editorial errors or omissions contained herein.185kg CO 2eq. kg CO2 Manufacturing77%Distribution 7% Use 16% End of Life 0% Value chain  carbon footprint  0% Value chain  carbon footprint
### Evidence:"""

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
