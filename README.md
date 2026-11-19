# Lanka News Lens

**The same story, across Sinhala, Tamil and English news, linked and summarised with citations.**

Sri Lankan news is published in three languages by outlets that rarely link to one another. Lanka News Lens
takes 136k articles from 14 outlets (2024–2026) and:

1. **Clusters articles into stories across languages** using bge-m3 embeddings and an online
   clustering pass. 10,605 stories are covered in more than one language, and 3,046 in all three.
2. **Classifies each story into 9 topics** with a small fine-tuned multilingual encoder. It is
   compared honestly against TF-IDF and keyword baselines, and the baseline wins.
3. **Writes a daily trilingual brief** (English, Sinhala, Tamil) with a 4B local LLM. Every
   sentence cites the articles it came from, and the citations are checked mechanically.
4. **Publishes a static site** with the briefs and a searchable story timeline:
   **https://nipuna-lakruwan.github.io/lanka-news-lens/**

The whole pipeline runs on an 8 GB MacBook Air M1. No cloud APIs are used.

> All data comes from Nuwan Senaratna's open news datasets,
> [nuuuwan/lk-news-docs](https://huggingface.co/datasets/nuuuwan/lk-news-docs) and
> [nuuuwan/lk-news-chunks](https://huggingface.co/datasets/nuuuwan/lk-news-chunks) (MIT), collected by
> [nuuuwan/lk_news](https://github.com/nuuuwan/lk_news). This project would not exist without that work.

## Results at a glance

| Part | What was measured | Result |
|---|---|---|
| Cross-language clustering | Precision of cross-language pairs in the same story (180 checked pairs) | **0.89** [0.84, 0.93] |
| Topic classifier (fine-tuned e5-small) | Macro-F1 on a 450-article checked test set | 0.715 [0.66, 0.76] |
| TF-IDF + logistic regression baseline | Same test set | **0.748** [0.69, 0.79] |
| Daily briefs | Sentences with valid citations / sentences judged faithful to the cited articles | 73/73 cite a real source; 51 supported, 10 minor slips, 12 wrong |

The numbers in square brackets are 95% confidence intervals. **Read [Limits and known errors](#limits-and-known-errors)
before you trust any of these numbers.** In particular, the "checked" labels were assigned by Claude (an
AI assistant), not by native-speaker annotators.

## Data

| | |
|---|---|
| Articles | 136,045 (Tamil 52,998 · English 51,306 · Sinhala 31,741) |
| Outlets | 14: Ada Derana (en/si), Ada, Daily Mirror, Daily FT, The Island, EconomyNext, Lankadeepa, Virakesari, Tamil Mirror, BBC Sinhala, Colombo Telegraph, dbsjeyaraj.com, News First |
| Period | January 2024 – November 2026 |
| Text used | title + the first body chunk |
| Revisions | pinned in [`scripts/download.sh`](scripts/download.sh) |

Sinhala coverage drops sharply in 2026, so most trilingual stories are from 2024–2025.

## 1. Cross-language story clustering

**Method.** Each article (its title plus the first 300 characters) is embedded with
[bge-m3](https://huggingface.co/BAAI/bge-m3) through Ollama. A single online pass in time order
([`cluster.py`](src/lanka_news_lens/cluster.py)), in the style of topic detection and tracking, does the rest:

- **Per-language mean centering.** Subtracting each language's mean vector removes most of the
  "this is Tamil" direction, so same-event pairs across languages score closer to same-language pairs.
- **Joining a story.** An article joins the most similar open story if the cosine similarity to that
  story's centroid is at least **0.55**. If the story does not yet contain the article's language,
  a **+0.03** bonus applies.
- **When a story closes.** A story closes after **36 h** without new articles, or **96 h** after its
  first article, whichever comes first. Without the 96 h cap, daily weather forecasts chained into
  "stories" that ran for months with 300+ articles.
- **Choosing the threshold.** The threshold was picked by reading cross-language nearest-neighbour
  pairs at several similarity levels ([`calibrate_pairs.py`](scripts/calibrate_pairs.py)).

| | |
|---|---|
| Stories | 90,378 (19,844 with ≥ 2 articles) |
| Multi-language stories | 10,605 |
| Trilingual stories | 3,046 |
| Largest story | 87 articles (polling-division results on 2024 election night) |
| Articles linked to another language | Sinhala 14,314 (45%) · Tamil 13,952 (26%) · English 13,061 (26%) |

**Precision evaluation.** I sampled 60 multi-language stories for each language pair, then drew one random article
per language from each story. Each pair was checked by asking whether the two articles report the same event
([`data/labels/cluster_pairs.csv`](data/labels/cluster_pairs.csv)).

| Pair | Same event | Precision [Wilson 95% CI] |
|---|---|---|
| English–Sinhala | 57 / 60 | 0.95 [0.86, 0.98] |
| English–Tamil | 54 / 60 | 0.90 [0.80, 0.95] |
| Sinhala–Tamil | 50 / 60 | 0.83 [0.72, 0.91] |
| **All** | **161 / 180** | **0.89 [0.84, 0.93]** |

The 19 errors fall into three groups:

- **Recurring items from neighbouring days.** A weather forecast or a fuel-price notice for one day is grouped with the next day's.
- **Follow-ups and reactions.** An arrest is grouped with a politician's reaction to the arrest.
- **Same-kind events.** Two different donations, two gas companies, or two singers' concerts.

Sinhala–Tamil is the weakest pair. These outlets
often run short wire-style items with few named entities.

**Recall was not measured.** Some same-event articles are surely split into separate stories.

## 2. Topic classifier

Nine labels: `politics`, `economy`, `crime`, `disaster_weather`, `health`, `sports`, `education`, `international`, `other`.

**Training labels are weak.** Only a few outlets put a topic section in their URLs, and those give 1,804 articles.
The rest come from trilingual keyword rules ([`weak_labels.py`](src/lanka_news_lens/weak_labels.py)), which give
18,372 articles and are wrong about 1 time in 5. The training set is capped at 760 articles per (language, topic).

**Test set.** 450 articles (150 per language) were drawn uniformly at random from 2024 onwards and labelled by reading each one
([`data/labels/topic_test.csv`](data/labels/topic_test.csv)). They are excluded from training.

| Model | All | English | Sinhala | Tamil | Accuracy |
|---|---|---|---|---|---|
| Keyword rules (the weak labeller) | 0.660 [0.60, 0.71] | 0.697 | 0.618 | 0.668 | 0.629 |
| **TF-IDF (word + char n-grams) + LR** | **0.748** [0.69, 0.79] | **0.777** | **0.761** | **0.672** | **0.771** |
| Fine-tuned multilingual-e5-small | 0.715 [0.66, 0.76] | 0.743 | 0.707 | 0.645 | 0.722 |

All scores are macro-F1, with 95% bootstrap CIs (1,000 resamples).

**The fine-tuned model does not beat TF-IDF.** Both learn from the same noisy labels, and TF-IDF's
character n-grams pick up the very keywords the labels were made from. The fine-tuned model's weakest class is
`other` (F1 0.45, recall 0.35), because the keyword rules rarely assign it. Its best class is `sports` (0.97).
The model is still published, because it is a single 118M-parameter encoder that handles all three scripts and is easy to reuse:
**[nipuna02/lk-news-topic-classifier](https://huggingface.co/nipuna02/lk-news-topic-classifier)**
(model card: [`model_card/README.md`](model_card/README.md)).

**Training ran on an 8 GB M1** (MPS, batch 16, 128 tokens, lr 5e-5). A 2-epoch run stopped partway through epoch 2
while the machine was shared with other jobs. The released weights are the end-of-epoch-1 checkpoint, which scored
macro-F1 0.917 on the weak-label validation split. The gap down to 0.715 on checked labels is mostly label noise,
so a second epoch would most likely have fitted that noise rather than closed the gap.

## 3. Cited trilingual daily briefs

For a given day, the brief takes the top 5 stories covered by at least two outlets, ranked by number of outlets + 2 × number of languages. Then ([`brief.py`](src/lanka_news_lens/brief.py)):

1. Up to 6 source articles per story, round-robin over languages with one per outlet, are numbered [1]…[n]. The model sees each title plus the first 700 characters of the body for English, or 350 for Sinhala and Tamil, which use more tokens per character.
2. **qwen3 4B Instruct** (`qwen3:4b-instruct-2507-q4_K_M` via Ollama, temperature 0.2) writes a headline and
   2–4 short English sentences. Every sentence must end with citations like `[1][3]`.
3. Each sentence is parsed. A sentence with no citation, or one that cites a number outside 1…n, is rejected.
   So is one that leaks Chinese, Korean or Cyrillic script. Up to two retries are made at a higher temperature, and the best attempt is kept.
4. Each English sentence is translated into Sinhala and Tamil, with the citations removed first and
   re-attached after, using **[NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)**.
   I tried letting qwen3 4B translate first. Its Sinhala looked fluent but often changed the facts.
   NLLB's tokenizer splits Sinhala conjuncts. It drops the zero-width joiner and prints ප් රකාශ for ප්‍රකාශ. The pieces are rejoined only when the joined word is in a list of 23k words from the corpus, so real word breaks such as විසින් රක්ෂිත are kept.

Example days (all on the site):

| Date | What happened |
|---|---|
| 2024-09-22 | Presidential election result |
| 2024-11-15 | Parliamentary election result |
| 2025-08-22 | Former President Ranil Wickremesinghe arrested |
| 2025-11-28 | Cyclone Ditwah |

Here is one story from **2025-08-22**, exactly as generated. The Sinhala and Tamil versions are NLLB translations of the English:

> **Sri Lanka’s law applies equally to all**
> Public Security Minister Ananda Wijepala stated everyone is equal before the law regardless of title or position. [1][2][3][4]
> Former President Ranil Wickremesinghe was arrested over alleged misuse of state funds. [4]
> He was remanded in custody by Colombo Magistrate’s Court and the case is now before the judge. [4]
>
> **ශ්‍රී ලංකා නීතිය සියලු දෙනාටම සමානව අදාළයි**
> මහජන ආරක්ෂක අමාත්‍ය ආනන්ද විජේපාල මහතා ප්‍රකාශ කළේ උසස් තනතුරක් හෝ තනතුරක් නොතකා නීතිය ඉදිරියේ සෑම කෙනෙකුම සමාන බවයි. [1][2][3][4] …
>
> **இலங்கை சட்டம் அனைவருக்கும் சமமாக பொருந்தும்**
> பொதுப் பாதுகாப்பு அமைச்சர் ஆனந்தா வைஜேபாலா, பதவி அல்லது பதவி எதுவாக இருந்தாலும் சட்டத்தின் முன் அனைவரும் சமம் என்று கூறினார். [1][2][3][4] …
>
> Sources:
> - [1] Ada Derana (en)
> - [2] Ada Derana Sinhala (si)
> - [3] Tamil Mirror (ta)
> - [4] The Island (en)

Even this good example shows the translation's limits. The Tamil renders the minister's name as "வைஜேபாலா", where Tamil outlets write விஜேபால, and it translates "title or position" as "பதவி அல்லது பதவி" ("position or position").

### How faithful are the briefs?

I read every English sentence (73 in all, over 20 stories) next to the articles it cites
([`scripts/brief_audit.py`](scripts/brief_audit.py), [`data/labels/brief_audit.csv`](data/labels/brief_audit.csv)).

| Day | Supported | Minor slip | Wrong |
|---|---|---|---|
| 2024-09-22 Presidential result | 17 | 0 | 3 |
| 2024-11-15 Parliamentary result | 8 | 3 | 7 |
| 2025-08-22 Ranil arrested | 15 | 1 | 0 |
| 2025-11-28 Cyclone Ditwah | 11 | 6 | 2 |
| **All** | **51 (70%)** | **10 (14%)** | **12 (16%)** |

The checker confirms that each citation points to a real source. It does not confirm that the source says what the sentence claims. The errors that remain have four causes:

1. **Names read from Sinhala or Tamil sources get mangled.** For example: "Navalpitikam" for Nawalapitiya, "Islam Muttu" for
   Ismail Muthu Mohamed, "Chairperson Nayanaka Rohan" for Secretary-General Kushani Rohanadeera, and "Janatha Vimukthi Party in
   Kali" for the New Democratic Front in Galle. A 4B model reads these scripts far less well than English.
2. **Updates within a story are not reconciled.** The NPP "won 123 seats" (a mid-count figure) sits beside "159 seats". The
   Ranil story leads with "summoned" although later sources in the same story say he was arrested. The curfew "until 6 a.m." had been extended to noon.
3. **Sources with titles only lead to invented details.** One 2024-11-15 story had four articles with empty bodies, and all four
   of its sentences are wrong, including invented district names. Since then, `make_briefs.py` skips body-less articles when there are enough
   others, but the published example still shows the failure.
4. **One meta-sentence** ("Source reports different vote counts…") slipped through as if it were news.

In short, the citations make every claim checkable, but they do not make the claims true. Treat these briefs as a starting point for reading the
sources, not as a substitute for them.

## 4. Site

[`site/`](site/) is plain HTML, CSS and JavaScript with no build step. It is deployed to GitHub Pages by
[`.github/workflows/pages.yml`](.github/workflows/pages.yml).

- **Briefs:** the example days in English, Sinhala and Tamil, with links from each citation to the source articles.
- **Timeline:** all 16,328 stories covered by two or more outlets, month by month.
  It has search in any script and filters by topic (from the classifier) and by language coverage.

## Limits and known errors

- **The "checked" labels are not human gold.** I built this project with Claude, an AI assistant, and Claude
  read and labelled the 450 test articles and the 180 cluster pairs. The labels were not made by native Sinhala
  or Tamil speakers. The numbers show the general level, not exact accuracy. Corrections are welcome as issues
  or PRs to [`data/labels/`](data/labels/).
- **Cluster recall is unknown.** Only precision was measured.
- **The fine-tuned classifier is worse than TF-IDF** on this test set, and it was trained for one epoch. Both
  models learn from weak labels that are wrong about 20% of the time.
- **The briefs can be wrong.** A 4B model reads short leads and does not reconcile updates within a story.
  Every sentence is traceable to a source, but see the audit above for the errors that still get through.
- **Machine translation.** The Sinhala and Tamil briefs are NLLB translations of the English brief, not
  summaries written natively in each language. Expect stiff phrasing and some transliterated names.
- **The coverage is uneven.** Sinhala thins out in 2026, and a few outlets dominate each language.
- **Article copyright stays with the outlets.** The site shows only titles and links back to the outlets.

## Reproduce

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) with:

```bash
ollama pull bge-m3
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Then:

```bash
git clone https://github.com/Nipuna-Lakruwan/lanka-news-lens && cd lanka-news-lens
uv sync
./scripts/download.sh                      # pinned dataset revisions, ~250 MB
uv run python scripts/prepare.py           # → data/processed/articles.parquet
uv run python scripts/embed.py             # bge-m3, resumable (~3.5 h on an M1)
uv run python scripts/run_clustering.py    # → stories
uv run python scripts/eval_cluster_pairs.py
uv run python scripts/build_trainset.py
uv run python scripts/baseline_tfidf.py
uv run python scripts/finetune.py --epochs 1
uv run python scripts/predict_topics.py
uv run python scripts/make_briefs.py 2024-09-22 2024-11-15 2025-08-22 2025-11-28
uv run python scripts/build_site.py        # → site/data/
uv run pytest
```

The checked label files are committed. The scripts that drew the samples (`make_testset.py`, `sample_cluster_pairs.py`)
are included for transparency, but rerunning them overwrites the samples.

## Licence and credit

- Code and the fine-tuned weights: MIT (see [LICENSE](LICENSE)).
- Data: Nuwan Senaratna, [nuuuwan/lk_news](https://github.com/nuuuwan/lk_news) (MIT). The copyright in the articles stays with the outlets.
- Models used:
  - [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT);
  - [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT);
  - [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0);
  - [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
    (**CC-BY-NC-4.0**, so the Sinhala and Tamil brief text is for non-commercial use).
