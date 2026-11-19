"""Generate cited trilingual briefs for the example days.

usage: python scripts/make_briefs.py 2024-09-22 2025-11-28 ...
Writes results/briefs/<day>.json (with the model's raw output for auditing).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from lanka_news_lens.brief import summarise_story, translate

TOP_K, MAX_SOURCES = 5, 6
OUTLETS = json.load(open("src/lanka_news_lens/outlets.json"))

arts = pd.read_parquet("data/processed/articles.parquet").merge(pd.read_parquet("data/processed/stories.parquet"))


def pick_sources(x: pd.DataFrame) -> pd.DataFrame:
    """Round-robin over languages (English first), one article per outlet where possible."""
    # Skip articles with no body: a title alone gave the model too little to go on (see the README audit).
    x = x[x.body.str.len() >= 80] if (x.body.str.len() >= 80).sum() >= 2 else x
    x = x.sort_values("time_ut").drop_duplicates("newspaper_id")
    by = {l: g for l, g in x.groupby("lang")}
    out, k = [], 0
    while len(out) < MAX_SOURCES and any(k < len(g) for g in by.values()):
        for l in ["en", "si", "ta"]:
            if l in by and k < len(by[l]) and len(out) < MAX_SOURCES:
                out.append(by[l].iloc[k])
        k += 1
    return pd.DataFrame(out)


def top_stories(day: str) -> list:
    d = arts[arts.date == day]
    g = d.groupby("story_id").agg(n=("doc_id", "size"), o=("newspaper_id", "nunique"), l=("lang", "nunique"))
    g["score"] = g.o + 2 * g.l
    g = g[g.o >= 2].sort_values(["score", "n"], ascending=False)
    return list(g.index[:TOP_K])


Path("results/briefs").mkdir(parents=True, exist_ok=True)
for day in sys.argv[1:]:
    t0 = time.time()
    stories = []
    for sid in top_stories(day):
        x = arts[arts.story_id == sid]
        src = pick_sources(x[x.date == day] if (x.date == day).sum() >= 2 else x)
        sources = [{"doc_id": r.doc_id, "title": r.title, "body": r.body, "lang": r.lang,
                    "outlet": OUTLETS.get(r.newspaper_id, r.newspaper_id), "url": r.url} for r in src.itertuples()]
        s = summarise_story(sources)
        if len(s["sentences"]) == 0:
            print("  skip story", sid, "(no cited sentences)")
            continue
        item = {"story_id": int(sid), "n_articles": int(len(x)), "n_outlets": int(x.newspaper_id.nunique()),
                "langs": sorted(x.lang.unique()),
                "sources": [{k: v for k, v in so.items() if k != "body"} for so in sources],
                "en": {"headline": s["headline"], "sentences": [{"text": t, "cites": c} for t, c in s["sentences"]]},
                "audit": {"raw": s["raw"], "rejected_sentences": s["rejected"], "attempts": s["attempts"]}}
        for lang in ["si", "ta"]:
            tr = translate([s["headline"]] + [t for t, _ in s["sentences"]], lang)
            if tr is None:
                print("  translation failed", sid, lang)
                continue
            item[lang] = {"headline": tr[0], "sentences": [{"text": t, "cites": c}
                                                           for t, (_, c) in zip(tr[1:], s["sentences"])]}
        stories.append(item)
        print(f"  {day} story {sid}: {len(s['sentences'])} sentences, rejected {s['rejected']}, "
              f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"date": day, "stories": stories}, open(f"results/briefs/{day}.json", "w"), ensure_ascii=False, indent=1)
