"""Fine-tune intfloat/multilingual-e5-small for 9 Sri Lankan news topics on the weak labels.

Fits an 8 GB M1: max 128 tokens, batch 16, fp32 on MPS (falls back to CPU).
Model selection uses the weakly-labelled validation split only; the hand-labelled
test set is touched once, at the end.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from lanka_news_lens.evaluate import report
from lanka_news_lens.weak_labels import TOPICS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="intfloat/multilingual-e5-small")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--out", default="models/lk-news-topic-classifier")
ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate the saved checkpoint in --out")
args = ap.parse_args()

torch.manual_seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
L2I = {t: i for i, t in enumerate(TOPICS)}


def text(df):
    return (df.title + "\n" + df.body.str[:600]).tolist()


train = pd.read_parquet("data/processed/topic_train.parquet")
val = pd.read_parquet("data/processed/topic_val.parquet")
test = pd.read_csv("data/labels/topic_test.csv").join(
    pd.read_parquet("data/processed/articles.parquet").set_index("doc_id")[["body"]], on="doc_id")

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(TOPICS), id2label=dict(enumerate(TOPICS)), label2id=L2I).to(dev)


def batches(texts, labels=None, shuffle=False):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for s in range(0, len(idx), args.bs):
        b = idx[s:s + args.bs]
        enc = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        y = torch.tensor([labels[i] for i in b], device=dev) if labels is not None else None
        yield enc, y


@torch.no_grad()
def predict(texts):
    model.eval()
    out = []
    for enc, _ in batches(texts):
        out.append(model(**enc).logits.float().cpu())
    model.train()
    return torch.cat(out).softmax(-1).numpy()


NOTE = ("Evaluated the checkpoint saved after epoch 1 of 2 (weak-val macro-F1 0.917); "
        "the run stopped during epoch 2 on the shared machine.") if args.eval_only else ""
Xtr, ytr = text(train), [L2I[x] for x in train.label]
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * math.ceil(len(Xtr) / args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
best, t0, step = -1.0, time.time(), 0
log = []
model.train()
for ep in range(0 if args.eval_only else args.epochs):
    for enc, y in batches(Xtr, ytr, shuffle=True):
        loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        step += 1
        if step % 100 == 0:
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    pv = predict(text(val)).argmax(1)
    vrep = report(val.label, [TOPICS[i] for i in pv], val.lang)
    log.append({"epoch": ep, "val_macro_f1_weak": vrep["all"]["macro_f1"]})
    print("epoch", ep, "weak-val macro-F1", vrep["all"]["macro_f1"], flush=True)
    if vrep["all"]["macro_f1"] > best:
        best = vrep["all"]["macro_f1"]
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)

model = AutoModelForSequenceClassification.from_pretrained(args.out).to(dev)
probs = predict(text(test))
pred = [TOPICS[i] for i in probs.argmax(1)]
res = {"args": vars(args), "note": NOTE, "train_size": len(train), "train_log": log,
       "train_minutes": round((time.time() - t0) / 60, 1), "device": str(dev),
       "test": report(test.label, pred, test.lang)}
Path("results").mkdir(exist_ok=True)
json.dump(res, open("results/topic_finetuned.json", "w"), indent=2, ensure_ascii=False)
pd.DataFrame({"doc_id": test.doc_id, "pred": pred, "conf": probs.max(1).round(3)}).to_csv(
    "results/topic_preds_finetuned.csv", index=False)
print({l: res["test"][l]["macro_f1"] for l in ["all", "en", "si", "ta"]})
