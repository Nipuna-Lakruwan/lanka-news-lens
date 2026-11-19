"""Write the static site's data: site/data/briefs.json and site/data/timeline/*.json."""
import json
from pathlib import Path

import pandas as pd

from lanka_news_lens.brief import fix_sinhala
from lanka_news_lens.weak_labels import TOPICS

OUT = Path("site/data")
(OUT / "timeline").mkdir(parents=True, exist_ok=True)
OUTLETS = json.load(open("src/lanka_news_lens/outlets.json"))
LABELS = json.load(open("results/brief_days.json"))

# briefs
days = []
for day, label in LABELS.items():
    b = json.load(open(f"results/briefs/{day}.json"))
    for s in b["stories"]:
        s.pop("audit", None)
        si = s.get("si")  # briefs made before fix_sinhala existed
        if si:
            si["headline"] = fix_sinhala(si["headline"])
            for x in si["sentences"]:
                x["text"] = fix_sinhala(x["text"])
    days.append({"date": day, "label": label, "stories": b["stories"]})
json.dump({"days": days}, open(OUT / "briefs.json", "w"), ensure_ascii=False, separators=(",", ":"))

# timeline: stories with >= 2 outlets
arts = pd.read_parquet("data/processed/articles.parquet").merge(pd.read_parquet("data/processed/stories.parquet"))
arts = arts.merge(pd.read_parquet("data/processed/article_topics.parquet"), on="doc_id")
g = arts.sort_values("time_ut").groupby("story_id")
stories = []
for sid, x in g:
    stories.append({"id": int(sid), "s": x.date.iloc[0], "e": x.date.iloc[-1], "l": sorted(x.lang.unique()),
                    "o": int(x.newspaper_id.nunique()), "tp": x.topic.mode().iloc[0],
                    "a": [[r.title, r.newspaper_id, r.lang, r.url] for r in x.itertuples()]})
df = pd.DataFrame({"m": [s["s"][:7] for s in stories]})
index = {"topics": TOPICS, "outlets": OUTLETS, "months": []}
for m in sorted(df.m.unique(), reverse=True):
    ss = [s for s in stories if s["s"][:7] == m]
    ss.sort(key=lambda s: s["s"], reverse=True)
    json.dump(ss, open(OUT / "timeline" / f"{m}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    index["months"].append({"m": m, "n": len(ss)})
json.dump(index, open(OUT / "timeline" / "index.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("stories on site:", len(stories), "months:", len(index["months"]))
