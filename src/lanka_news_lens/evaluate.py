"""Per-language macro-F1 with bootstrap 95% confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .weak_labels import TOPICS


def _macro(y, p):
    labels = sorted(set(y))  # classes present in this slice of the gold labels
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def boot_ci(y, p, n=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = [_macro(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def report(y_true, y_pred, langs) -> dict:
    y_true, y_pred, langs = map(np.asarray, (y_true, y_pred, langs))
    out = {}
    for lang in ["all", "en", "si", "ta"]:
        m = np.ones(len(y_true), bool) if lang == "all" else langs == lang
        y, p = y_true[m], y_pred[m]
        out[lang] = {
            "n": int(m.sum()),
            "macro_f1": round(float(_macro(y, p)), 3),
            "macro_f1_ci95": boot_ci(y, p),
            "accuracy": round(float(accuracy_score(y, p)), 3),
        }
    out["per_class"] = {
        k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")} | {"support": int(v["support"])}
        for k, v in classification_report(y_true, y_pred, labels=TOPICS, output_dict=True, zero_division=0).items()
        if k in TOPICS
    }
    return out
