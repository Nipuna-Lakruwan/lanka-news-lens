"""Draw the topic test set: a uniform random sample of 150 articles per language
from 2024-01 onwards (the period where all three languages are present).

These doc_ids are excluded from classifier training. Labels are written by hand
into data/labels/topic_test.csv (column ``label``); see README for who checked them.
"""
from pathlib import Path

import pandas as pd

df = pd.read_parquet("data/processed/articles.parquet")
df = df[(df.date >= "2024-01-01") & (df.body.str.len() > 50)]
s = df.groupby("lang").sample(150, random_state=2026)
s = s[["doc_id", "lang", "newspaper_id", "date", "title", "body"]].copy()
s["lead"] = s.body.str[:260].str.replace("\n", " ")
s = s.drop(columns="body").sample(frac=1, random_state=1)
Path("data/labels").mkdir(parents=True, exist_ok=True)
s.to_csv("data/labels/topic_test_unlabelled.csv", index=False)
print(s.lang.value_counts())
