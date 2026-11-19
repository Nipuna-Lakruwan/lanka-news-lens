"""Print cross-language nearest-neighbour pairs at different similarity levels so the
clustering threshold can be chosen by reading them (uses whatever shards exist)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lanka_news_lens.cluster import center_by_language

shards = sorted(Path("data/processed/emb").glob("shard-*.npy"))
emb = np.concatenate([np.load(p) for p in shards]).astype(np.float32)
df = pd.read_parquet("data/processed/articles.parquet").iloc[: len(emb)]
m = (df.date >= "2024-01-01").to_numpy()
df, emb = df[m].reset_index(drop=True), emb[m]
X = center_by_language(emb, df.lang.to_numpy())
raw = emb / np.linalg.norm(emb, axis=1, keepdims=True)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
rows = []
t = df.time_ut.to_numpy()
for i in rng.choice(np.where(df.lang == "en")[0], 600, replace=False):
    for tl in ["si", "ta"]:
        cand = np.where((df.lang == tl).to_numpy() & (np.abs(t - t[i]) < 36 * 3600))[0]
        if not len(cand):
            continue
        s = X[cand] @ X[i]
        j = cand[s.argmax()]
        rows.append((float(s.max()), float(raw[j] @ raw[i]), df.title[i], df.title[j], tl))
r = pd.DataFrame(rows, columns=["sim", "raw", "en", "other", "tl"])
r["bin"] = pd.cut(r.sim, [0, .45, .5, .55, .6, .65, .7, .75, 1])
print(r.groupby("bin", observed=True).size())
for b, g in r.groupby("bin", observed=True):
    print("\n##", b)
    for _, x in g.sample(min(6, len(g)), random_state=0).iterrows():
        print(f"{x.sim:.2f} raw{x.raw:.2f} [{x.tl}] {x.en[:70]}  <->  {x.other[:70]}")
