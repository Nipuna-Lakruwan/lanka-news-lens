"""Join docs with their first body chunk, clean text, write data/processed/articles.parquet."""
import urllib.parse
from pathlib import Path

import pandas as pd

from lanka_news_lens.text import clean

RAW = Path("data/raw")
docs = pd.read_parquet(RAW / "docs.parquet")
chunks = pd.concat(
    [pd.read_parquet(p, columns=["doc_id", "chunk_index", "chunk_text"]) for p in sorted(RAW.glob("chunks-*.parquet"))]
)
body = (
    chunks.sort_values(["doc_id", "chunk_index"])
    .groupby("doc_id")
    .agg(body=("chunk_text", lambda s: "\n\n".join(s)), n_chunks=("chunk_index", "size"))
    .reset_index()
)
df = docs.merge(body, on="doc_id", how="left")
df["title"] = df.description.map(clean)
df["body"] = df.body.fillna("").map(clean)
# first chunk repeats the title as its first line; drop it from the body
first_line = df.body.str.split("\n", n=1).str[0]
df.loc[first_line == df.title, "body"] = df.body.str.split("\n", n=1).str[1].fillna("")
df["body"] = df.body.str.strip()
df["url"] = df.url_metadata
df["section"] = df.url.map(lambda u: urllib.parse.unquote(urllib.parse.urlparse(u).path).strip("/").split("/")[0])
df["date"] = df.date_str
df = df[["doc_id", "date", "time_ut", "lang", "newspaper_id", "title", "body", "url", "section", "n_chunks"]]
df = df.drop_duplicates("doc_id").sort_values("time_ut").reset_index(drop=True)
print(df.shape, "with body:", (df.body.str.len() > 50).mean().round(3))
print(df.groupby("lang").apply(lambda g: (g.body.str.len() > 50).mean()).round(3))
df.to_parquet("data/processed/articles.parquet", index=False)
