import numpy as np
from rouge_score import rouge_scorer

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def rouge_scores(predictions, references):
    assert len(predictions) == len(references)
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    per_example = {t: [] for t in ROUGE_TYPES}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for t in ROUGE_TYPES:
            per_example[t].append(scores[t].fmeasure)


    return {t: np.mean(per_example[t]) for t in ROUGE_TYPES}