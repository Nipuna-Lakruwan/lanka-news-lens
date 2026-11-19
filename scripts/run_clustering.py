"""Cluster all articles into stories and write data/processed/stories.parquet + story table."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lanka_news_lens.cluster import center_by_language, cluster

THRESHOLD, CROSS_BONUS, WINDOW_H, MAX_SPAN_H = 0.55, 0.03, 36.0, 96.0

df = pd.read_parquet("data/processed/articles.parquet")
emb = np.concatenate([np.load(p) for p in sorted(Path("data/processed/emb").glob("shard-*.npy"))])
assert len(emb) == len(df), (len(emb), len(df))
keep = (df.date >= "2024-01-01").to_numpy()  # before 2024 there are only ~250 stray articles
df, emb = df[keep].reset_index(drop=True), emb[keep]
X = center_by_language(emb, df.lang.to_numpy())
t0 = time.time()
df["story_id"] = cluster(X, df.time_ut.to_numpy(), df.lang.to_numpy(), THRESHOLD, CROSS_BONUS, WINDOW_H, MAX_SPAN_H)
print(f"clustered {len(df)} articles in {time.time() - t0:.0f}s")

g = df.groupby("story_id")
stories = pd.DataFrame({
    "n_articles": g.size(),
    "n_outlets": g.newspaper_id.nunique(),
    "langs": g.lang.agg(lambda s: "".join(sorted(set(s)))),
    "start": g.date.min(),
    "end": g.date.max(),
})
stories["n_langs"] = stories.langs.str.len() // 2
print(stories.n_articles.describe())
print("stories with >=2 articles:", (stories.n_articles >= 2).sum())
print("multi-language stories:", (stories.n_langs >= 2).sum(), " trilingual:", (stories.n_langs == 3).sum())
art_in_multi = df.story_id.map(stories.n_langs) >= 2
print("articles in multi-language stories, by lang:", df[art_in_multi].lang.value_counts().to_dict(),
      "of", df.lang.value_counts().to_dict())
df[["doc_id", "story_id"]].to_parquet("data/processed/stories.parquet", index=False)
stories.to_parquet("data/processed/story_table.parquet")
json.dump({"threshold": THRESHOLD, "cross_bonus": CROSS_BONUS, "window_h": WINDOW_H, "max_span_h": MAX_SPAN_H, "n_articles": len(df),
           "n_stories": int(len(stories)), "n_multi_article": int((stories.n_articles >= 2).sum()),
           "n_multilang": int((stories.n_langs >= 2).sum()), "n_trilingual": int((stories.n_langs == 3).sum()),
           "articles_in_multilang_by_lang": {k: int(v) for k, v in df[art_in_multi].lang.value_counts().items()},
           "articles_by_lang": {k: int(v) for k, v in df.lang.value_counts().items()}},
          open("results/clustering_stats.json", "w"), indent=2)
