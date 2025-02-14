"""Cosql response accuracy metric."""

import datasets
from typing import Dict, Any


def compute_sacrebleu_metric(predictions, references) -> Dict[str, Any]:
    sacrebleu = datasets.load_metric("sacrebleu")
    references = [[r["utterances"]] for r in references]
    results = sacrebleu.compute(predictions=predictions, references=references)
    score = round(results["score"], 2)
    return {
        "sacrebleu": float(score),
    }