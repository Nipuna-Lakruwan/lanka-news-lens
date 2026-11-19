"""Sentence-level faithfulness check of the example briefs (English text against the cited articles).
Verdicts were assigned by reading each sentence next to its cited sources (by Claude, not a native speaker):
  ok     - supported by the cited sources
  minor  - substantively right, but a name/office is garbled or a detail overstated
  wrong  - not supported, or contradicted, by the cited sources
Writes data/labels/brief_audit.csv and results/brief_audit.json."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ISSUES = {  # (day, story index, sentence index): (verdict, note); everything else is "ok"
    ("2024-09-22", 2, 3): ("wrong", "meta-comment about sources, not news"),
    ("2024-09-22", 4, 0): ("wrong", "curfew was extended to 12 noon; 10pm-6am was the original curfew"),
    ("2024-09-22", 4, 3): ("wrong", "source mentions the 9th presidential election, not a president's decision"),
    ("2024-11-15", 0, 0): ("minor", "123 seats was a mid-count figure; final was 159 (stated in the next sentence)"),
    ("2024-11-15", 0, 1): ("wrong", "conflates the two-thirds claim with Gampaha's 16 seats"),
    ("2024-11-15", 2, 0): ("wrong", "sources had titles only; 'Kaludi District' is invented (source is Badulla)"),
    ("2024-11-15", 2, 1): ("wrong", "'Kallarikai' is invented; source is Galle district, Akmeemana results"),
    ("2024-11-15", 2, 2): ("wrong", "Bandaragama is in Kalutara, not 'Kallarikai'"),
    ("2024-11-15", 2, 3): ("wrong", "source 3 IS the Matale result"),
    ("2024-11-15", 3, 3): ("minor", "vague paraphrase of 'MPs who represented parliament for decades'"),
    ("2024-11-15", 4, 0): ("minor", "names garbled (Manusha Nanayakkara, Ramesh Pathirana)"),
    ("2024-11-15", 4, 1): ("wrong", "they ran for the New Democratic Front in Galle, not 'Janatha Vimukthi Party' in 'Kali'"),
    ("2024-11-15", 4, 2): ("wrong", "garbled party name; the front won no seat in Galle"),
    ("2025-08-22", 2, 2): ("minor", "one STF officer was injured, not several"),
    ("2025-11-28", 0, 1): ("minor", "it is the Irrigation Department"),
    ("2025-11-28", 1, 2): ("minor", "sources say officers who cannot reach their station may report to the nearest one"),
    ("2025-11-28", 2, 0): ("minor", "name garbled (Ismail Muthu Mohamed)"),
    ("2025-11-28", 2, 2): ("wrong", "letter went to Secretary-General Kushani Rohanadeera, not 'Chairperson Nayanaka Rohan'"),
    ("2025-11-28", 3, 0): ("minor", "place name garbled (Nawalapitiya)"),
    ("2025-11-28", 3, 2): ("minor", "Nawalapitiya divisional hospital"),
    ("2025-11-28", 3, 3): ("wrong", "source says 5 deaths so far in the area, not 'the fifth death'"),
    ("2025-11-28", 4, 0): ("minor", "storm centre was off the east coast (south of Trincomalee)"),
}
STORY_NOTES = {
    ("2025-08-22", 0): "headline and lead say 'summoned'; later sources in the same story report the arrest, which the brief omits",
}

rows = []
for day in json.load(open("results/brief_days.json")):
    for si, s in enumerate(json.load(open(f"results/briefs/{day}.json"))["stories"]):
        for j, x in enumerate(s["en"]["sentences"]):
            v, note = ISSUES.get((day, si, j), ("ok", ""))
            rows.append({"date": day, "story_id": s["story_id"], "story": si, "sentence": j,
                         "text": x["text"], "cites": " ".join(map(str, x["cites"])), "verdict": v, "note": note})
df = pd.DataFrame(rows)
df.to_csv("data/labels/brief_audit.csv", index=False)
c = Counter(df.verdict)
out = {"sentences": len(df), **{k: c[k] for k in ["ok", "minor", "wrong"]},
       "by_day": {d: dict(Counter(g.verdict)) for d, g in df.groupby("date")}, "story_notes":
       {f"{d} story {i}": n for (d, i), n in STORY_NOTES.items()}}
Path("results/brief_audit.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
