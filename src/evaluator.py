from evaluate import load
from typing import List, Dict

class Evaluator:
    def __init__(self):
        self.bleu = load('bleu')
        self.rouge = load('rouge')
        self.ppl = load('perplexity')

    def compute_automated(self, preds: List[str], refs: List[str]) -> Dict:
        results = {}
        try:
            results['BLEU'] = self.bleu.compute(predictions=preds, references=[[r] for r in refs])
        except Exception as e:
            results['BLEU_error'] = str(e)
        try:
            results['ROUGE'] = self.rouge.compute(predictions=preds, references=refs)
        except Exception as e:
            results['ROUGE_error'] = str(e)
        # Perplexity compute requires model logits; placeholder
        return results

    def human_eval_table(self, samples):
        # samples: list of dicts with keys input, response, score, notes
        lines = ["| Input | Model Response | Human Score (1-5) | Notes |","|---|---|---:|---|" ]
        for s in samples:
            lines.append(f"| {s.get('input','')} | {s.get('response','')} | {s.get('score','')} | {s.get('notes','')} |")
        return "\n".join(lines)
