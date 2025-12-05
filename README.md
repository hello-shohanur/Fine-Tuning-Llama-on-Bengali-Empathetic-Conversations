# Bengali Empathetic Conversation Fine-Tuning
Fine-tuning **LLaMA 3.1-8B-Instruct** on the Bengali Empathetic Conversations Corpus using LoRA/Unsloth.

## Overview
This repository contains the pipeline for fine-tuning LLaMA 3.1-8B-Instruct to generate empathetic Bengali responses. The project uses parameter-efficient fine-tuning (LoRA) and supports Unsloth for 4-bit optimized training on constrained GPUs (e.g. Kaggle).

## Repository Structure
```
bengali-empathetic-llama-finetuning/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_finetuning_llama_lora.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── dataset_processor.py
│   ├── fine_tuner.py
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── lora_strategy.py
│   │   └── unsloth_strategy.py
│   ├── evaluator.py
│   └── utils.py
├── logs/
├── checkpoints/
├── requirements.txt
└── LICENSE
```

## Quickstart
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Preprocess dataset (open `notebooks/01_data_preprocessing.ipynb`).

3. Run fine-tuning (open `notebooks/02_finetuning_llama_lora.ipynb`).

4. Evaluate (open `notebooks/03_evaluation.ipynb`).

## Contact
AI Engineer — [Your Company Name]
