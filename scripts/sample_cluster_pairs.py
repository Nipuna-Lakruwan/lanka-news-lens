"""Sample cross-language article pairs that the clustering put in the same story,
for hand-checking: 60 en-si, 60 en-ta, 60 si-ta. Writes data/labels/cluster_pairs_unlabelled.csv."""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/articles.parquet").merge(pd.read_parquet("data/processed/stories.parquet"))
rng = np.random.default_rng(11)
rows = []
for a, b in [("en", "si"), ("en", "ta"), ("si", "ta")]:
    g = df[df.lang.isin([a, b])].groupby("story_id")
    ok = [sid for sid, x in g if x.lang.nunique() == 2]
    for sid in rng.choice(ok, 60, replace=False):
        x = df[df.story_id == sid]
        ra = x[x.lang == a].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        rb = x[x.lang == b].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        rows.append({"pair": f"{a}-{b}", "story_id": sid, "story_size": len(x),
                     "doc_a": ra.doc_id, "date_a": ra.date, "title_a": ra.title, "lead_a": ra.body[:200].replace("\n", " "),
                     "doc_b": rb.doc_id, "date_b": rb.date, "title_b": rb.title, "lead_b": rb.body[:200].replace("\n", " ")})
pd.DataFrame(rows).to_csv("data/labels/cluster_pairs_unlabelled.csv", index=False)
print(len(rows))
