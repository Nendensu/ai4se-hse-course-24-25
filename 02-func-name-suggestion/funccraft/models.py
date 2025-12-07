from collections.abc import Iterable
from functools import cache
from pprint import pprint

import datasets
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))


def predict(dataset: datasets.Dataset, model_name: str, task: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    predictions = []
    references = []

    for row in tqdm(dataset):
        if task == 1:
            input_text = row["body_no_comments"]
        else:
            input_text = row["body_with_comments"]

        if not input_text.strip():
            predictions.append("")
            references.append(row["extracted_name"])
            continue

        input_text = f"def <extra_id_0>():\n{input_text}"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=20)

        pred_name = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_name = pred_name.strip()
        if pred_name:
            pred_name = pred_name.split()[0].split("(")[0].split(":")[0]
        else:
            pred_name = ""

        predictions.append(pred_name)
        references.append(row["extracted_name"])

    eval_results = run_evaluate(predictions=predictions, references=references)

    print("\n" + "*" * 80)
    print("Evaluation results:")
    pprint(eval_results)
    print("*" * 80 + "\n")


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
