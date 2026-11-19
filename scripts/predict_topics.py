"""Run the fine-tuned classifier over every article in a multi-outlet story.
Writes data/processed/article_topics.parquet (doc_id, topic, conf)."""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

M = "models/lk-news-topic-classifier"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
arts = pd.read_parquet("data/processed/articles.parquet").merge(pd.read_parquet("data/processed/stories.parquet"))
n_out = arts.groupby("story_id").newspaper_id.nunique()
arts = arts[arts.story_id.map(n_out) >= 2].reset_index(drop=True)
texts = (arts.title + "\n" + arts.body.str[:600]).tolist()
probs = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
        probs.append(model(**enc).logits.float().softmax(-1).cpu().numpy())
        if i % 6400 == 0:
            print(i, len(texts), flush=True)
p = np.concatenate(probs)
id2 = model.config.id2label
pd.DataFrame({"doc_id": arts.doc_id, "topic": [id2[int(k)] for k in p.argmax(1)], "conf": p.max(1).round(3)}).to_parquet(
    "data/processed/article_topics.parquet", index=False)
