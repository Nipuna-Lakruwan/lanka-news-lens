"""Attach hand labels (data/labels/parts/topic_*.txt, whitespace-separated, in row order)
to the sampled test articles and write data/labels/topic_test.csv."""
from pathlib import Path

import pandas as pd

from lanka_news_lens.weak_labels import TOPICS

s = pd.read_csv("data/labels/topic_test_unlabelled.csv")
labels = []
for p in sorted(Path("data/labels/parts").glob("topic_*.txt")):
    labels += p.read_text().split()
assert len(labels) == len(s), (len(labels), len(s))
assert set(labels) <= set(TOPICS), set(labels) - set(TOPICS)
s["label"] = labels
s.to_csv("data/labels/topic_test.csv", index=False)
print(pd.crosstab(s.label, s.lang, margins=True))
