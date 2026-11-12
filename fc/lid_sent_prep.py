"""Language ID (16 Latin-script languages incl. look-alike pairs) and binary review sentiment items.
LID main: FLORES-200 (mteb/flores, CC-BY-SA-4.0); unseen: MASSIVE utterances (CC-BY-4.0).
Sentiment main: amazon_polarity (Apache-2.0); unseen: IMDB (test) + SST-2 (validation)."""
import json, random, collections
import pandas as pd
LANGS = [("Spanish", "spa_Latn", "es-ES"), ("Portuguese", "por_Latn", "pt-PT"), ("Italian", "ita_Latn", "it-IT"), ("French", "fra_Latn", "fr-FR"),
         ("Romanian", "ron_Latn", "ro-RO"), ("Danish", "dan_Latn", "da-DK"), ("Norwegian", "nob_Latn", "nb-NO"), ("Swedish", "swe_Latn", "sv-SE"),
         ("Dutch", "nld_Latn", "nl-NL"), ("Afrikaans", "afr_Latn", "af-ZA"), ("German", "deu_Latn", "de-DE"), ("Indonesian", "ind_Latn", "id-ID"),
         ("Malay", "zsm_Latn", "ms-MY"), ("Polish", "pol_Latn", "pl-PL"), ("Slovenian", "slv_Latn", "sl-SL"), ("Icelandic", "isl_Latn", "is-IS")]
LSYS = ("Identify the language of the text the user sends. Possible languages: " + ", ".join(l for l, _, _ in LANGS) +
        ".\nReply with the language name only, exactly as written above.")
SSYS = "Classify the sentiment of the review the user sends as positive or negative.\nReply with positive or negative only."
def it(i, src, cat, gold, sys_, text):
    return {"id": f"{src}-{i}", "src": src, "cat": cat, "gold": gold, "tools": None,
            "messages": [{"role": "system", "content": sys_}, {"role": "user", "content": text}]}
rng = random.Random(20261111)
fl = pd.concat([pd.read_parquet("data_raw/lid/flores_dev.parquet"), pd.read_parquet("data_raw/lid/flores_devtest.parquet")], ignore_index=True)
lid_main = [it(f"{c}-{i}", "flores", f"flores/{l}", l.lower(), LSYS, fl[c][i]) for l, c, _ in LANGS for i in range(len(fl))]
lid_unseen = []
for l, _, loc in LANGS:
    rows = [json.loads(x) for x in open(f"data_raw/lid/massive_{loc}.jsonl")]
    lid_unseen += [it(f"{loc}-{r['id']}", "massive", f"massive/{l}", l.lower(), LSYS, r["text"]) for r in rows]
rng.shuffle(lid_main); rng.shuffle(lid_unseen)
amz = pd.read_parquet("data_raw/sent/amz_test.parquet").sample(n=6000, random_state=1)
s_main = [it(i, "amazon", f"amazon/{'positive' if r.label else 'negative'}", "positive" if r.label else "negative", SSYS, (r.title + "\n" + r.content)[:3000]) for i, r in amz.iterrows()]
imdb = pd.read_parquet("data_raw/sent/imdb_test.parquet").sample(n=1500, random_state=1); sst = pd.read_parquet("data_raw/sent/sst2_val.parquet")
s_unseen = [it(i, "imdb", "imdb", "positive" if r.label else "negative", SSYS, r.text[:3000]) for i, r in imdb.iterrows()] + \
           [it(i, "sst2", "sst2", "positive" if r.label else "negative", SSYS, r.sentence) for i, r in sst.iterrows()]
rng.shuffle(s_main); rng.shuffle(s_unseen)
out = {"lid_calib": lid_main[:1000], "lid_dev": lid_main[1000:2600], "lid_unseen_probe": lid_unseen[:1600],
       "sent_calib": s_main[:1000], "sent_dev": s_main[1000:2000], "sent_unseen_probe": s_unseen[:1000]}
def take(lst, catn, cap):
    c = collections.Counter(); sel, rest = [], []
    for x in lst:
        (sel if c[x["cat"]] < cap else rest).append(x); c[x["cat"]] += 1
    return sel, rest
lid_exam_m, lid_pool = take(lid_main[2600:], None, 300)
lid_exam_u, _ = take(lid_unseen[1600:], None, 300)
amz2 = pd.read_parquet("data_raw/sent/amz_test.parquet").drop(amz.index).sample(n=6000, random_state=2)
s_pool = [it(i, "amazon", f"amazon/{'positive' if r.label else 'negative'}", "positive" if r.label else "negative", SSYS, (r.title + "\n" + r.content)[:3000]) for i, r in amz2.iterrows()]
out.update({"lid_exam": lid_exam_m + lid_exam_u, "lid_pool": lid_pool[:8000],
            "sent_exam": s_main[2000:4000] + s_unseen[1000:], "sent_pool": s_pool})
for k, v in out.items():
    with open(f"data/{k}.jsonl", "w") as f:
        for x in v: f.write(json.dumps(x) + "\n")
    print(k, len(v), len(collections.Counter(x["cat"] for x in v)))
