"""Embed every article (title + first ~300 chars of body) with bge-m3 via Ollama.

Resumable: writes data/processed/emb/shard-XXXX.npy (float16, L2-normalised), 2,048 rows each.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("data/processed/emb")
OUT.mkdir(parents=True, exist_ok=True)
SHARD, BATCH = 2048, 32
URL = "http://localhost:11434/api/embed"


def embed_text(df: pd.DataFrame) -> list[str]:
    return (df.title + "\n" + df.body.str[:300]).tolist()


def embed(texts: list[str]) -> np.ndarray:
    for attempt in range(5):
        try:
            r = requests.post(URL, json={"model": "bge-m3", "input": texts, "truncate": True,
                                         "options": {"num_ctx": 256}}, timeout=900)
            r.raise_for_status()
            e = np.asarray(r.json()["embeddings"], dtype=np.float32)
            return e / np.linalg.norm(e, axis=1, keepdims=True)
        except Exception as ex:  # noqa: BLE001
            print("retry", attempt, ex, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("embedding failed")


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/articles.parquet")
    texts = embed_text(df)
    n_shards = (len(df) + SHARD - 1) // SHARD
    t0 = time.time()
    for s in range(n_shards):
        f = OUT / f"shard-{s:04d}.npy"
        if f.exists():
            continue
        chunk = texts[s * SHARD:(s + 1) * SHARD]
        vecs = np.concatenate([embed(chunk[i:i + BATCH]) for i in range(0, len(chunk), BATCH)])
        np.save(f, vecs.astype(np.float16))
        print(f"shard {s + 1}/{n_shards} done, {time.time() - t0:.0f}s", flush=True)
    sys.exit(0)
