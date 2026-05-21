#!/usr/bin/env python3
"""
corpus_fetch.py  --  build the paper corpus for the demo knowledge base.

Pulls full-text articles from the PubMed Central Open Access subset for one
gene, and keeps ONLY the commercial-use licences (CC0 / CC BY). The PMC OA
subset also contains a non-commercial tier; do not put that in a demo you
show on a public stage.

Output:  ./corpus/PMCxxxxxxx.txt   (one plain-text article per file)
Then:    aws s3 sync ./corpus s3://<your-bucket>/corpus/

The corpus itself is NOT committed to this repo -- run this to build it.

NCBI etiquette: <=3 requests/sec without an API key, <=10 with one.
Set NCBI_API_KEY to go faster:  https://www.ncbi.nlm.nih.gov/account/

Requires: requests
"""

import os
import time
import xml.etree.ElementTree as ET

import requests

GENE = "PCSK9"
EXTRA = "(LDL OR cholesterol OR cardiovascular)"
TARGET = 1000
OUTDIR = "corpus"
API_KEY = os.environ.get("NCBI_API_KEY", "")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
COMMERCIAL_LICENCES = {"CC0", "CC BY"}  # safe for a public talk
DELAY = 0.12 if API_KEY else 0.34


def _key():
    return {"api_key": API_KEY} if API_KEY else {}


def search_pmc(retmax):
    term = f'{GENE} AND {EXTRA} AND "open access"[filter]'
    ids, retstart = [], 0
    while len(ids) < retmax:
        r = requests.get(
            f"{EUTILS}/esearch.fcgi",
            params={
                "db": "pmc",
                "term": term,
                "retmax": 200,
                "retstart": retstart,
                "retmode": "json",
                **_key(),
            },
        )
        r.raise_for_status()
        batch = r.json()["esearchresult"]["idlist"]
        if not batch:
            break
        ids += batch
        retstart += len(batch)
        time.sleep(DELAY)
    return ids[:retmax]


def fetch_article(pmcid):
    """Return (licence, full_text). full_text is None if unusable."""
    r = requests.get(
        f"{EUTILS}/efetch.fcgi", params={"db": "pmc", "id": pmcid, "retmode": "xml", **_key()}
    )
    if r.status_code != 200 or not r.content:
        return None, None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None, None

    licence = "UNKNOWN"
    lic_el = root.find(".//permissions/license")
    if lic_el is not None:
        blob = (
            lic_el.get("{http://www.w3.org/1999/xlink}href", "") + " " + "".join(lic_el.itertext())
        ).lower()
        if "publicdomain" in blob or "cc0" in blob:
            licence = "CC0"
        elif "creativecommons.org/licenses/by/" in blob or "cc by" in blob:
            licence = "CC BY"
        elif "/by-nc" in blob or "noncommercial" in blob:
            licence = "CC BY-NC"
    if licence not in COMMERCIAL_LICENCES:
        return licence, None

    body = root.find(".//body")
    if body is None:
        return licence, None
    text = " ".join(t.strip() for t in body.itertext() if t.strip())
    return (licence, text) if len(text) > 1500 else (licence, None)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Searching PMC for open-access {GENE} articles...")
    ids = search_pmc(int(TARGET * 1.8))  # over-fetch; many get filtered out
    print(f"  {len(ids)} candidate PMCIDs")

    kept, skipped = 0, 0
    for pmcid in ids:
        if kept >= TARGET:
            break
        licence, text = fetch_article(pmcid)
        time.sleep(DELAY)
        if text is None:
            skipped += 1
            continue
        with open(f"{OUTDIR}/PMC{pmcid}.txt", "w", encoding="utf-8") as f:
            f.write(f"# PMC{pmcid}  licence: {licence}\n\n{text}")
        kept += 1
        if kept % 50 == 0:
            print(f"  kept {kept}  (skipped {skipped} non-commercial / empty)")

    print(f"\nDone. {kept} commercial-use articles in ./{OUTDIR}/")
    print(f"Next:  aws s3 sync ./{OUTDIR} s3://YOUR-BUCKET/corpus/")


if __name__ == "__main__":
    main()
