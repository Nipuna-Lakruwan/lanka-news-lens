"""Precision of cross-language matches on the checked sample (data/labels/cluster_pairs.csv)."""
import json
import math

import pandas as pd

# Judgements on the 180 sampled pairs: index -> same event? (checked by Claude, see README)
NOT_SAME = {
    11: "related: Kim arrives in Beijing vs. the parade itself (next day)",
    18: "follow-up: clearance service suspended vs. hotlines two days later",
    35: "weather forecasts for different days",
    81: "different events: Maldives president's call vs. Pakistan PM's condolences",
    91: "Navy's yearly drug total vs. a single 150 kg seizure",
    102: "reaction (Chandrika's statement) vs. the arrest itself, two days apart",
    112: "SLFP backs abolishing presidency vs. a remark about the SJB",
    116: "restaurant prices raised vs. cut three days later",
    119: "two different singers' deaths",
    121: "match result vs. tournament preview",
    126: "two different court cases against Meta",
    134: "weather forecasts for different days",
    145: "two different donors to the same fund",
    147: "22 women MPs vs. 18 doctor MPs",
    152: "Vijay launches party vs. Namal congratulating him",
    159: "weather forecasts for different days",
    162: "election complaint counts from different days",
    170: "two different politicians backing Ranil",
    179: "Laugfs vs. Litro gas price cuts",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


df = pd.read_csv("data/labels/cluster_pairs_unlabelled.csv")
df["same_event"] = [i not in NOT_SAME for i in range(len(df))]
df["note"] = [NOT_SAME.get(i, "") for i in range(len(df))]
df.to_csv("data/labels/cluster_pairs.csv", index=False)
out = {}
for name, g in [("all", df)] + list(df.groupby("pair")):
    k, n = int(g.same_event.sum()), len(g)
    out[name] = {"n": n, "same_event": k, "precision": round(k / n, 3), "ci95_wilson": wilson(k, n)}
json.dump(out, open("results/cluster_eval.json", "w"), indent=2)
print(json.dumps(out, indent=1))
