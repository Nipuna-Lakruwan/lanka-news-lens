---
license: mit
language:
- si
- ta
- en
library_name: transformers
pipeline_tag: text-classification
base_model: intfloat/multilingual-e5-small
datasets:
- nuuuwan/lk-news-docs
tags:
- sri-lanka
- sinhala
- tamil
- news
- topic-classification
metrics:
- f1
---

# lk-news-topic-classifier

A small topic classifier for Sri Lankan news in **Sinhala, Tamil and English**. It is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(118M parameters) fine-tuned on 9 topics:

`politics`, `economy`, `crime`, `disaster_weather`, `health`, `sports`, `education`, `international`, `other`

It is part of [Lanka News Lens](https://github.com/Nipuna-Lakruwan/lanka-news-lens), which
clusters the same story across Sinhala, Tamil and English outlets and writes cited daily briefs.

> **Honest headline:** on the checked test set this model does **not** beat a TF-IDF +
> logistic-regression baseline (macro-F1 0.715 vs 0.748; the confidence intervals overlap).
> Use it if you want a single small multilingual encoder. If you only need the best accuracy on this label set, the TF-IDF baseline in the repo is as good or better.

## Results

The test set is 450 articles (150 per language), sampled at random from 2024–2026 and labelled
by hand. Macro-F1 is averaged over the classes present. The 95% CIs come from 1,000 bootstrap
resamples.

| Model | All | English | Sinhala | Tamil |
|---|---|---|---|---|
| Keyword rules (the weak labeller) | 0.660 [0.60, 0.71] | 0.697 | 0.618 | 0.668 |
| TF-IDF (word + char) + logistic regression | **0.748** [0.69, 0.79] | **0.777** | **0.761** | **0.672** |
| **This model** | 0.715 [0.66, 0.76] | 0.743 | 0.707 | 0.645 |

Accuracy of this model: 0.722 overall.

Per-class F1 for this model:

| Class | F1 | Support |
|---|---|---|
| sports | 0.97 | 18 |
| disaster_weather | 0.79 | 32 |
| politics | 0.79 | 104 |
| economy | 0.77 | 76 |
| crime | 0.76 | 80 |
| international | 0.71 | 43 |
| education | 0.62 | 8 |
| health | 0.57 | 10 |
| other | 0.45 | 79 |

The weakest class is `other` (recall 0.35): the keyword rules that produced the training labels
rarely assign it, so the model learned to push borderline articles into a named topic.
Education and health have only 8 and 10 test articles, so their scores are very noisy.

## How it was trained

- **Data:** [nuuuwan/lk-news-docs](https://huggingface.co/datasets/nuuuwan/lk-news-docs) and
  [nuuuwan/lk-news-chunks](https://huggingface.co/datasets/nuuuwan/lk-news-chunks) (MIT), collected
  by Nuwan Senaratna in [nuuuwan/lk_news](https://github.com/nuuuwan/lk_news). The model sees the
  title plus the first 600 characters of the body, truncated to 128 tokens.
- **Labels are weak.** Most outlets have no topic sections in their URLs, so the training labels come from:
  - URL sections, where they exist (ada.lk, Daily Mirror, Tamil Mirror and some Lankadeepa sections): 1,804 articles;
  - trilingual keyword rules (Sinhala/Tamil stems, English whole words): 18,372 articles.

  A spot check suggests about 1 in 5 of these weak labels is wrong. Training uses 20,176
  articles, capped at 760 per (language, topic). Sinhala `other` (525) and English `sports`
  (651) fall below the cap. Test articles were excluded.
- **Hardware:** an 8 GB MacBook Air M1 (MPS), shared with other jobs. Batch size 16, learning rate 5e-5, AdamW
  with a linear schedule and 6% warmup.
- **Training length:** the run was planned for 2 epochs but stopped partway through the
  second. The released weights are the checkpoint saved after epoch 1, which scored macro-F1
  0.917 on the weak-label validation split. The gap between 0.917 on weak labels and 0.715 on
  checked labels is mostly label noise.

## Who checked the test labels

The 450 test labels were assigned by Claude, an AI assistant, reading each article. They were
**not** labelled by a native-speaker human. Treat the numbers as indicative, and open an issue if
you find label errors. The labelling conventions:
- road and other accidents → `other`;
- foreign events → `international`;
- Sri Lanka's bilateral relations → `politics`;
- relief work for floods and landslides → `disaster_weather`.

## Use

```python
from transformers import pipeline
clf = pipeline("text-classification", model="nipuna02/lk-news-topic-classifier")
clf("ශ්‍රී ලංකා මහ බැංකුව ප්‍රතිපත්ති පොලී අනුපාත නොවෙනස්ව තබයි")
```

## Limits

- The model only knows Sri Lankan outlets from 2024–2026, and only the 9 topics above.
- Every article gets exactly one label, even when it mixes topics (for example, an economic
  policy debate in parliament).
- Sinhala coverage in the source data is thin from 2026 onwards.
- Not for moderation or any decision about people.

## Licence and credit

The weights are released under MIT. The base model is `intfloat/multilingual-e5-small` (MIT). The news data is
by Nuwan Senaratna (MIT). The copyright in the articles stays with the outlets.
