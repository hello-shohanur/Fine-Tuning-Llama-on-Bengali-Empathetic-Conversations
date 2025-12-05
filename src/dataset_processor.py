import json
from datasets import Dataset
from typing import List

class DatasetProcessor:
    def __init__(self, tokenizer=None, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_csv(self, path: str):
        import pandas as pd
        df = pd.read_csv(path)
        return df

    def to_instruction_format(self, df, context_col='context', response_col='response'):
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "instruction": "একটি সহানুভূতিশীল উত্তর লিখুন।",
                "input": str(row.get(context_col, "")),
                "output": str(row.get(response_col, ""))
            })
        return samples

    def to_hf_dataset(self, samples: List[dict]):
        return Dataset.from_list(samples)

    def tokenize_batch(self, samples):
        if not self.tokenizer:
            raise ValueError("Tokenizer not set on DatasetProcessor")
        # combine instruction + input as prompt, keep full sequence
        inputs = [s['instruction'] + "\n" + s['input'] for s in samples]
        targets = [s['output'] for s in samples]
        enc = self.tokenizer(inputs, truncation=False, max_length=self.max_length, padding='longest', return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, truncation=False, max_length=self.max_length, padding='longest', return_tensors='pt')
        enc['labels'] = labels['input_ids']
        return enc
