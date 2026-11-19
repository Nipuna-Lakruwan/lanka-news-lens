"""Write src/lanka_news_lens/si_cluster_words.txt.gz, the Sinhala words with an internal hal cluster (e.g. ප්‍ර, ක්ෂ) seen at least 3 times in the corpus.
fix_sinhala uses it to tell a conjunct that NLLB split ("ප් රකාශ") from a real word break ("විසින් රක්ෂිත")."""
import collections
import gzip
import re

import pandas as pd

a = pd.read_parquet("data/processed/articles.parquet", columns=["lang", "title", "body"])
a = a[a.lang == "si"]
word = re.compile(r"[඀-෿‍]+")
cluster = re.compile("්[ක-ෆ]")
c = collections.Counter()
for t in a.title + " " + a.body:
    for w in word.findall(t):
        w = w.replace("‍", "")
        if cluster.search(w):
            c[w] += 1
words = sorted(w for w, n in c.items() if n >= 3)
with gzip.open("src/lanka_news_lens/si_cluster_words.txt.gz", "wt", encoding="utf-8") as f:
    f.write("\n".join(words))
print(len(words), "words")
