"""Cited trilingual daily briefs.

For one day we take the biggest cross-outlet stories and ask a local Ollama model
(qwen3 4B) for a short English summary, one sentence per line, in which *every
sentence ends with citation markers* such as ``[2]`` or ``[1][3]``. The markers
are checked mechanically. The summary is then translated into Sinhala and Tamil
one sentence at a time with NLLB-200 (distilled 600M): the markers are stripped
before translation and re-attached afterwards, so each translated sentence keeps
exactly the citations of the English sentence it came from.

Why not let qwen3 4B translate? We tried. Its Sinhala was fluent-looking
nonsense and its Tamil got titles wrong ("Speaker" became "parliament leader").

Sentences that fail the check after the retries are dropped, never shown uncited.
"""
from __future__ import annotations

import re

import requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LANG_NAME = {"en": "English", "si": "Sinhala", "ta": "Tamil"}
CITE = re.compile(r"((?:\s*\[\d+\])+)")


def chat(messages, temperature=0.2, num_predict=400, num_ctx=4096) -> str:
    r = requests.post(OLLAMA, json={"model": MODEL, "messages": messages, "stream": False,
                                    "options": {"temperature": temperature, "num_predict": num_predict,
                                                "num_ctx": num_ctx, "seed": 7}}, timeout=1800)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def format_sources(sources) -> str:
    lines = []
    for k, s in enumerate(sources, 1):
        lead = s["body"][: 700 if s["lang"] == "en" else 350].replace("\n", " ")
        lines.append(f"[{k}] ({s['outlet']}, {LANG_NAME[s['lang']]}) {s['title']}\n{lead}")
    return "\n\n".join(lines)


FOREIGN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0400-\u04ff]")  # CJK, Hangul, Cyrillic leaks


def split_cited(text: str, n_sources: int):
    """One sentence per line. Return [(sentence_without_markers, [ids])] and the rejected count."""
    out, rejected = [], 0
    for line in text.splitlines():
        sent = line.strip().lstrip("-*• ").strip()
        if len(sent) < 8:
            continue
        if FOREIGN.search(sent):
            rejected += 1
            continue
        ids = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
        clean = CITE.sub("", sent).strip()
        clean = re.sub(r"\s+([.!?])$", r"\1", clean)
        if ids and all(1 <= i <= n_sources for i in ids):
            out.append((clean, sorted(set(ids))))
        else:
            rejected += 1
    return out, rejected


SUMMARY_PROMPT = """You are writing one item of a neutral daily news brief about Sri Lanka.
Below are news articles from different outlets (some in Sinhala or Tamil) about ONE story.

Write a headline (max 12 words) on the first line, starting with "HEADLINE: ".
Then write 2 to 4 short factual sentences in English that summarise the story, each sentence on its own line.
Rules:
- Use only facts stated in the sources. No opinions, no background you were not given.
- End EVERY sentence with the number(s) of the source(s) that support it, like [1] or [2][4].
- Prefer facts reported by more than one source, and cite all of them.

SOURCES:
{sources}
"""

def summarise_story(sources, retries=2):
    best = None
    for attempt in range(retries + 1):
        raw = chat([{"role": "user", "content": SUMMARY_PROMPT.format(sources=format_sources(sources))}],
                   temperature=0.2 + 0.2 * attempt)
        m = re.search(r"HEADLINE:\s*(.+)", raw)
        headline = CITE.sub("", m.group(1)).strip().strip('"*') if m else ""
        if FOREIGN.search(headline):
            headline = ""
        body = raw[m.end():] if m else raw
        sents, rejected = split_cited(body, len(sources))
        cand = {"headline": headline, "sentences": sents, "rejected": rejected, "attempts": attempt + 1, "raw": raw}
        if best is None or (len(sents), -rejected) > (len(best["sentences"]), -best["rejected"]):
            best = cand
        if headline and len(sents) >= 2 and rejected == 0:
            break
    return best


_NLLB = {}
_SI_WORDS: list[str] = []
_SI_CORE = re.compile("[\u0d80-\u0dff\u200d]+")
_SI_CORE_END = re.compile("[\u0d80-\u0dff\u200d]+$")


def _si_known_prefix(stem: str) -> bool:
    """True if `stem` (ZWJ removed) is a word, or the start of a word, from the Sinhala corpus."""
    if not _SI_WORDS:
        import gzip
        from pathlib import Path
        with gzip.open(Path(__file__).with_name("si_cluster_words.txt.gz"), "rt", encoding="utf-8") as f:
            _SI_WORDS.extend(f.read().split("\n"))
        _SI_WORDS.append("")  # keeps bisect in range
        _SI_WORDS.sort()
    import bisect
    i = bisect.bisect_left(_SI_WORDS, stem)
    return i < len(_SI_WORDS) and _SI_WORDS[i].startswith(stem)


def fix_sinhala(text: str) -> str:
    """Rejoin consonant clusters that NLLB's tokenizer splits.

    NLLB drops the zero-width joiner and inserts a space, printing "ප් රකාශය" for ප්‍රකාශය or "ආරක් ෂක" for
    ආරක්ෂක. It happens only before ර, ය and ෂ. A hal (්) before a space is also a normal word ending ("විසින් රක්ෂිත"), so the pieces are only
    joined when the result starts a word seen in the Sinhala corpus (si_cluster_words.txt.gz).
    """
    out: list[str] = []
    for tok in text.split(" "):
        prev = out[-1] if out else ""
        left = _SI_CORE_END.search(prev)
        right = _SI_CORE.match(tok)
        if (prev.endswith("\u0dca") and left and right and tok[0] in "\u0dbb\u0dba\u0dc2"
                and _si_known_prefix((left.group() + right.group()).replace("\u200d", ""))):
            out[-1] = prev + ("\u200d" if tok[0] in "\u0dbb\u0dba" else "") + tok
        else:
            out.append(tok)
    return " ".join(out)


NLLB_CODE = {"si": "sin_Sinh", "ta": "tam_Taml"}


def translate(lines: list[str], lang: str) -> list[str] | None:
    """English -> Sinhala/Tamil with facebook/nllb-200-distilled-600M (CPU, beam 4)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not _NLLB:
        name = "facebook/nllb-200-distilled-600M"
        _NLLB["tok"] = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
        _NLLB["model"] = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    tok, model = _NLLB["tok"], _NLLB["model"]
    import torch

    with torch.no_grad():
        enc = tok(lines, return_tensors="pt", padding=True, truncation=True, max_length=200)
        out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODE[lang]),
                             num_beams=4, max_new_tokens=256)
    res = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
    if lang == "si":
        res = [fix_sinhala(t) for t in res]
    return res if all(res) else None
