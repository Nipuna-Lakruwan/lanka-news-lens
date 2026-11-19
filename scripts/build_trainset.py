"""Apply weak labels to every article, drop the hand-labelled test articles,
cap each (language, topic) cell, and write train/val splits."""
import pandas as pd

from lanka_news_lens.weak_labels import weak_label

CAP = 800
df = pd.read_parquet("data/processed/articles.parquet")
test_ids = set(pd.read_csv("data/labels/topic_test.csv").doc_id)
df = df[~df.doc_id.isin(test_ids) & (df.body.str.len() > 50)].copy()
res = [weak_label(*r) for r in zip(df.newspaper_id, df.section, df.title, df.body, df.lang)]
df["label"] = [r[0] for r in res]
df["label_source"] = [r[1] for r in res]
df = df[df.label.notna()]
print("weakly labelled:", len(df))
df = df.groupby(["lang", "label"]).sample(frac=1, random_state=0).groupby(["lang", "label"]).head(CAP)
df = df.sample(frac=1, random_state=0)
val = df.groupby(["lang", "label"]).head(40)
train = df.drop(val.index)
cols = ["doc_id", "lang", "newspaper_id", "date", "title", "body", "label", "label_source"]
train[cols].to_parquet("data/processed/topic_train.parquet", index=False)
val[cols].to_parquet("data/processed/topic_val.parquet", index=False)
print(pd.crosstab(train.label, train.lang, margins=True))
print(train.label_source.value_counts())
