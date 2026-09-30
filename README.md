# CarbonLLM

Code for [CarbonPDF](https://doi.org/10.1145/3744255.3798128), a question-answering method that extracts product carbon footprints from sustainability reports. These scripts fine-tune Llama 3 8B, run few-shot prompts on other models, retrieve relevant product text with TF-IDF, and score numeric answers.

- Paper: [Extracting Product Carbon Footprint in PDF Documents using Question Answering Framework](https://dl.acm.org/doi/10.1145/3744255.3798128) (E-Energy '26)
- Dataset paper: [An electronic product carbon footprint dataset for question answering](https://www.nature.com/articles/s41597-026-06544-5) (*Scientific Data*, 2026)
- Dataset: [CarbonPDF on AWS](https://registry.opendata.aws/carbonpdf/) and [pittcps/carbonpdf-dataset](https://github.com/pittcps/carbonpdf-dataset)

## Layout

```
src/
  finetune/      QLoRA fine-tuning of Llama 3 8B
  evaluation/    Inference with fine-tuned checkpoints, plus RMSE / MAE / exact match
  fewshot/       Few-shot runs (GPT-4o, Gemini, DeepSeek)
  retriever/     TF-IDF embeddings and top-k product retrieval
  input/         CSVs the scripts read (not included)
  output/        Generated predictions, embeddings, and logs
  model/         LoRA checkpoints loaded at evaluation time
```

Run each script from its own directory. Paths such as `../input/train.csv` are relative to that working directory.

## Data

Questions and product text come from [CarbonPDF-QA](https://registry.opendata.aws/carbonpdf/) (CC BY 4.0). The dataset and collection scripts are in [pittcps/carbonpdf-dataset](https://github.com/pittcps/carbonpdf-dataset). Place the CSVs this repo expects under `src/input/`. They are not included here.

| File | Used by | Columns |
| --- | --- | --- |
| `train.csv` | Fine-tuning | `Question`, `Text`, `Ground truth answer`, `Program` |
| `test.csv` | Fine-tuning eval, Llama and DeepSeek inference, Gemini with a reference | `Question`, `Text` |
| `test_relevant.csv` | Gemini without a reference, retrieval | `Question`, `Product ID` |
| `products_text.csv` | Retrieval | `Product ID`, `Product name`, `Text` |

Answers are Python lists of numbers, or lists of single-key dicts such as `[{'display': 37.2}]`. The baseline fine-tune target is a Python program in `Program` whose `answer` variable is that list.

## Setup

Use a CUDA machine. Llama 3 8B needs a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B`.

```bash
pip install torch transformers peft bitsandbytes datasets accelerate trl \
  pandas numpy scikit-learn matplotlib google-generativeai python-dotenv
```

For Gemini, set `GOOGLE_API_KEY` in a `.env` file next to the few-shot scripts (they call `load_dotenv()`).

## Fine-tuning

Both scripts load Llama 3 8B in 4-bit (NF4) and train a LoRA adapter (`r=8`, `lora_alpha=16`, dropout `0.05`) for 10 epochs.

From `src/finetune/`:

```bash
python baseline.py      # model writes a Python program that sets `answer`
python no_program.py    # model writes the answer list directly
```

Checkpoints are written to `src/finetune/llama3-8b-baseline/` and `src/finetune/llama3-8b-no-program/`.

## Evaluation

Copy or point the evaluation scripts at a saved adapter:

- `src/evaluation/eval_baseline.py` loads `../model/checkpoint-3600-baseline-Feb11` and executes the generated program.
- `src/evaluation/eval_no_program.py` loads `../model/checkpoint-1800-llama` and parses a list after `### Answer:`.

Each script retries generation up to 5 times, then writes `N/A`. Outputs land in `src/output/`.

`src/evaluation/metrics.py` reads `src/output/llama_fewshot_replaced.csv` (`Ground truth answer`, `Predicted answer`) and prints RMSE, MAE, and exact-match rate. Rows whose RMSE is above 40,000 are printed and scored as zero.

## Few-shot

From `src/fewshot/`:

| Script | Model | Input |
| --- | --- | --- |
| `gpt4o_jsonl.py` | Builds an OpenAI Batch file for `gpt-4o-2024-11-20` | `../output/test.csv` |
| `gemini.py` | `gemini-1.5-flash`, question only | `../input/test_relevant.csv` |
| `gemini_RAG.py` | `gemini-2.0-flash`, question plus reference text | `../input/test.csv` |
| `deepseek.py` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` in 4-bit | `../input/test.csv` |

`gpt4o_jsonl.py` only writes the JSONL request file. Submit that file with the OpenAI Batch API separately.

## Retrieval

From `src/retriever/`:

```bash
python TFIDF_embedding.py   # fit TF-IDF on products_text.csv, save ../output/tfidf_embeddings.pkl
python retrieve_topK.py     # top-10 product texts per question, recall printed at the end
```

`retrieve_topK.py` writes `src/output/test_relevant_top10.csv` and reports how often the ground-truth product is inside the top 10.

## Citation

If you use this code, please cite the E-Energy paper. If you use the CarbonPDF-QA data, please also cite the *Scientific Data* paper.

```bibtex
@inproceedings{zhao2026carbonpdf,
  author = {Zhao, Kaiwen and Balaji, Bharathan and Lee, Stephen},
  title = {Extracting Product Carbon Footprint in PDF Documents using Question Answering Framework},
  year = {2026},
  isbn = {9798400720116},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3744255.3798128},
  doi = {10.1145/3744255.3798128},
  booktitle = {Proceedings of the 17th ACM International Conference on Future and Sustainable Energy Systems},
  pages = {584--596},
  numpages = {13},
  keywords = {Information retrieval, Question answering},
  series = {E-Energy '26}
}

@article{zhao2026carbonpdfqa,
  author = {Zhao, Kaiwen and Koyatan Chathoth, Ajesh and Balaji, Bharathan and Lee, Stephen},
  title = {An electronic product carbon footprint dataset for question answering},
  journal = {Scientific Data},
  year = {2026},
  volume = {13},
  number = {1},
  pages = {228},
  doi = {10.1038/s41597-026-06544-5},
  url = {https://doi.org/10.1038/s41597-026-06544-5}
}
```
