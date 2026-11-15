"""RULER prompts and scoring (Hsieh et al., 2024), as the paper uses them.

A prompt file is the JSON-lines output of RULER's own `prepare.py` (see `scripts/make_ruler_prompts.sh`): each row has
`input` (the prompt in the model's chat template), `answer_prefix` (the first words of the answer, appended to the
prompt) and `outputs` (the reference strings). Accuracy is RULER's string match.
"""
import json

# new tokens RULER generates per task family (its scripts/data/synthetic/constants.py)
TOKENS_TO_GENERATE = {"niah": 128, "variable_tracking": 30, "common_words_extraction": 120, "freq_words_extraction": 50, "qa": 32}


def read_prompts(path, n=None, offset=0):
    """Rows [offset, offset + n) of a RULER prompt file."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[offset:] if n is None else rows[offset:offset + n]
    if n is not None and len(rows) < n:
        raise ValueError(f"{path} has {len(rows)} rows from row {offset}, {n} asked")
    return rows


def prompt_text(row):
    return row["input"] + row.get("answer_prefix", "")


def string_match_all(prediction, references):
    """RULER's string_match_all for one prompt, in [0, 1]: the fraction of the references found in the prediction."""
    p = prediction.lower()
    return sum(1.0 if r.lower() in p else 0.0 for r in references) / len(references)


def string_match_part(prediction, references):
    """RULER's string_match_part for one prompt: 1 when any reference is found in the prediction."""
    p = prediction.lower()
    return max(1.0 if r.lower() in p else 0.0 for r in references)


def accuracy(predictions, references, metric=string_match_all):
    """Mean score over the prompts, in percent, rounded as RULER rounds it."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("one prediction per reference list")
    return round(100.0 * sum(metric(p, r) for p, r in zip(predictions, references)) / len(predictions), 2)
