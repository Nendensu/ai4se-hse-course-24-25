from pathlib import Path

import datasets
import pandas as pd

import re
import copy
from .util import URL_PATTERN, SWEAR_PATTERNS, CONTRACTIONS


def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()

    # delete URL
    text = re.sub(URL_PATTERN, "", text)

    # delete contract
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)

    # remove repeat
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # swear pattern subs
    for swear, patterns in SWEAR_PATTERNS.items():
        for pat in patterns:
            text = re.sub(pat, f' {swear} ', text)

    # special symb delete
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # unnecessary whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare(raw_data: Path) -> datasets.Dataset:
    df = pd.read_excel(raw_data)

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['message', 'is_toxic'], inplace=True)
    df['message_clean'] = df['message'].apply(clean_text)

    return datasets.Dataset.from_pandas(df)


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
