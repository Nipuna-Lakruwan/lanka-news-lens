"""Baselines on the hand-labelled test set:
(a) the keyword rules themselves (unmatched -> 'other'),
(b) TF-IDF (word + char n-grams) + logistic regression trained on the weak labels."""
import json

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from lanka_news_lens.evaluate import report
from lanka_news_lens.text import tokens
from lanka_news_lens.weak_labels import weak_label

train = pd.read_parquet("data/processed/topic_train.parquet")
test = pd.read_csv("data/labels/topic_test.csv")
arts = pd.read_parquet("data/processed/articles.parquet").set_index("doc_id")
test = test.join(arts[["body", "section"]], on="doc_id")


def text(df):
    return (df.title + "\n" + df.body.str[:600]).tolist()


rules_pred = [weak_label(n, s, t, b, l)[0] or "other"
              for n, s, t, b, l in zip(test.newspaper_id, test.section, test.title, test.body, test.lang)]
out = {"keyword_rules": report(test.label, rules_pred, test.lang)}

word = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                       min_df=2, max_features=200_000, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000, sublinear_tf=True)
Xtr = hstack([word.fit_transform(text(train)), char.fit_transform(text(train))]).tocsr()
Xte = hstack([word.transform(text(test)), char.transform(text(test))]).tocsr()
clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
clf.fit(Xtr, train.label)
pred = clf.predict(Xte)
out["tfidf_logreg"] = report(test.label, pred, test.lang)
pd.DataFrame({"doc_id": test.doc_id, "pred": pred}).to_csv("results/topic_preds_tfidf.csv", index=False)
json.dump(out, open("results/topic_baselines.json", "w"), indent=2, ensure_ascii=False)
for k, v in out.items():
    print(k, {l: v[l]["macro_f1"] for l in ["all", "en", "si", "ta"]})
