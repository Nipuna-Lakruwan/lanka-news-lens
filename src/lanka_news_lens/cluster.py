"""Online cross-language story clustering over bge-m3 embeddings.

Articles are processed in publication order. Each one joins the most similar
*active* story (one that received an article in the last ``window_h`` hours) if
the cosine similarity to that story's centroid is at least ``threshold``;
otherwise it starts a new story. This is the classic single-pass event-detection
scheme from TDT, with two changes for a trilingual corpus:

* **Language centering.** bge-m3 vectors carry a language component, so two
  unrelated Sinhala articles can look closer than a Sinhala and an English
  article about the same event. We subtract each language's mean vector and
  re-normalise, which removes most of that offset.
* **Cross-language bonus.** A story that already has no article in the incoming
  article's language is easier to join (``cross_bonus`` is subtracted from the
  threshold), because same-language near-duplicates are the easy case and
  translations are the case we care about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def center_by_language(emb: np.ndarray, langs: np.ndarray) -> np.ndarray:
    out = emb.astype(np.float32).copy()
    for lang in np.unique(langs):
        m = langs == lang
        out[m] -= out[m].mean(0, keepdims=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


@dataclass
class Story:
    sid: int
    vec_sum: np.ndarray
    last_t: float
    first_t: float
    members: list = field(default_factory=list)
    langs: set = field(default_factory=set)

    @property
    def centroid(self):
        return self.vec_sum / (np.linalg.norm(self.vec_sum) + 1e-9)


def cluster(emb: np.ndarray, times: np.ndarray, langs: np.ndarray, threshold=0.62,
            cross_bonus=0.04, window_h=36.0, max_span_h=96.0) -> np.ndarray:
    """Return a story id per row. ``emb`` must already be centred and normalised.

    ``max_span_h`` closes a story that many hours after its first article, even if
    it is still receiving articles. Without it, recurring items such as the daily
    weather forecast chain into one "story" that lasts for months.
    """
    order = np.argsort(times, kind="stable")
    sid = np.full(len(emb), -1, dtype=np.int64)
    active: list[Story] = []
    next_id = 0
    win = window_h * 3600
    span = max_span_h * 3600
    C = np.zeros((0, emb.shape[1]), np.float32)
    dirty = True
    for i in order:
        t = times[i]
        if active and (t - min(s.last_t for s in active) > win or t - min(s.first_t for s in active) > span):
            active = [s for s in active if t - s.last_t <= win and t - s.first_t <= span]
            dirty = True
        if dirty:
            C = np.stack([s.centroid for s in active]) if active else np.zeros((0, emb.shape[1]), np.float32)
            dirty = False
        best, bj = -1.0, -1
        if len(C):
            sims = C @ emb[i]
            # stories missing this language get a lower bar
            bonus = np.array([cross_bonus if langs[i] not in s.langs else 0.0 for s in active], np.float32)
            adj = sims + bonus
            bj = int(adj.argmax())
            best = float(adj[bj])
        if best >= threshold:
            s = active[bj]
            s.vec_sum = s.vec_sum + emb[i]
            s.last_t = t
            s.members.append(int(i))
            s.langs.add(langs[i])
            sid[i] = s.sid
            C[bj] = s.centroid
        else:
            s = Story(next_id, emb[i].copy(), t, t, [int(i)], {langs[i]})
            next_id += 1
            active.append(s)
            sid[i] = s.sid
            dirty = True
    return sid
