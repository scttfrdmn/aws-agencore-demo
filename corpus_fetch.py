#!/usr/bin/env python3
"""
corpus_fetch.py  --  build the paper corpus for the demo knowledge base.

What this does:
  Searches PubMed Central (PMC) for open-access PCSK9 papers, downloads the
  full text of each one as XML via the NCBI E-utilities API, parses out the
  article body, and saves each as a plain-text .txt file in ./corpus/.

  After running this, sync the corpus to S3 before running build_kb.py:
      aws s3 sync ./corpus s3://<your-bucket>/corpus/

Why we filter by licence:
  The PMC Open Access subset has two tiers:
    - Commercial-use licences (CC0, CC BY): safe to use in a public demo.
    - Non-commercial licences (CC BY-NC, etc.): NOT allowed in a commercial
      or public presentation without the author's permission.
  This script keeps only CC0 and CC BY articles.  The result is a smaller
  corpus (~650 papers instead of ~2,000) but one you can safely show on stage.

Output:
  ./corpus/PMCxxxxxxx.txt   one plain-text article per file.
  Each file starts with a comment header:
      # PMCxxxxxxx  licence: CC0

  The corpus/ directory is gitignored.  Re-run this script to refresh it;
  files already present are overwritten (idempotent).

What to do if the fetch fails:
  - Rate limit (429 or HTTP errors): add NCBI_API_KEY (see below) to go faster,
    or reduce TARGET, or just wait a few minutes.
  - ParseError on some articles: normal -- PMC XML is inconsistent.  The script
    skips unparseable articles and continues.
  - "connection refused" / timeout: check your internet connection; NCBI is
    occasionally slow under load.  The script will resume from where it left
    off (by design: it fetches IDs first, then articles one by one).

NCBI API key (optional, recommended):
  Without an API key, NCBI limits you to 3 requests/second.
  With an API key (free): 10 requests/second -- cuts fetch time by ~3×.
  Get one at: https://www.ncbi.nlm.nih.gov/account/
  Then set it before running:
      export NCBI_API_KEY=your-key-here

Requires: requests (installed via uv pip install -e ".[dev]")
"""

import os
import time
import xml.etree.ElementTree as ET

import requests

# The gene we're searching for.  All three demo questions are about PCSK9.
GENE = "PCSK9"

# Co-terms to narrow the search to PCSK9's most relevant literature.
# Without this, "PCSK9" alone returns too many tangential papers.
EXTRA = "(LDL OR cholesterol OR cardiovascular)"

# How many articles to keep in the final corpus.
# We fetch more candidates than this (see over-fetch below) because many
# articles will be filtered out (non-commercial licence, no body text, too short).
TARGET = 1000

# Where to write the .txt files.
OUTDIR = "corpus"

# NCBI API key -- optional, but strongly recommended for faster fetching.
API_KEY = os.environ.get("NCBI_API_KEY", "")

# Base URL for NCBI E-utilities (the API that powers PubMed and PMC).
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Only these licences are safe for a public commercial demo.
COMMERCIAL_LICENCES = {"CC0", "CC BY"}

# Delay between requests -- NCBI rate limit is 3/sec without a key, 10/sec with.
# We leave a small margin (0.12 instead of 0.1) to avoid 429 errors.
DELAY = 0.12 if API_KEY else 0.34


def _key():
    """Return API key parameter dict, or empty dict if no key is set."""
    return {"api_key": API_KEY} if API_KEY else {}


def search_pmc(retmax):
    """Search PMC for open-access PCSK9 articles and return a list of PMC IDs.

    Uses the NCBI ESearch API to retrieve PMC IDs (integers) matching the
    search term.  Paginates in batches of 200 until retmax IDs are collected.

    Args:
        retmax: maximum number of IDs to return.

    Returns:
        A list of PMC ID strings (e.g. ["13156736", "5481105", ...]).
    """
    term = f'{GENE} AND {EXTRA} AND "open access"[filter]'
    ids, retstart = [], 0
    while len(ids) < retmax:
        r = requests.get(
            f"{EUTILS}/esearch.fcgi",
            params={
                "db": "pmc",
                "term": term,
                "retmax": 200,  # fetch 200 IDs per page
                "retstart": retstart,  # offset for pagination
                "retmode": "json",
                **_key(),
            },
        )
        r.raise_for_status()
        batch = r.json()["esearchresult"]["idlist"]
        if not batch:
            break  # no more results
        ids += batch
        retstart += len(batch)
        time.sleep(DELAY)
    return ids[:retmax]


def fetch_article(pmcid):
    """Fetch one PMC article, check its licence, and return its plain text.

    Downloads the article XML via the NCBI EFetch API, parses the licence
    from the <permissions> element, and extracts the body text.

    Articles are skipped (returning None for text) if:
      - The HTTP response is not 200 or the body is empty.
      - The XML cannot be parsed.
      - The licence is not CC0 or CC BY.
      - There is no <body> element (some PMC records have abstract-only entries).
      - The extracted text is shorter than 1,500 characters (too thin to be useful).

    Args:
        pmcid: the numeric PMC ID string (without the "PMC" prefix).

    Returns:
        (licence, text) where text is the plain string body, or (licence, None)
        if the article should be skipped.
    """
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "retmode": "xml", **_key()},
    )
    if r.status_code != 200 or not r.content:
        return None, None

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        # PMC XML is occasionally malformed.  Skip and move on.
        return None, None

    # Extract and classify the licence from the <permissions> section.
    # PMC uses href attributes and plain text in varying combinations,
    # so we concatenate both and search for known patterns.
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
            licence = "CC BY-NC"  # non-commercial -- skip

    # Filter out anything that is not explicitly commercial-use.
    if licence not in COMMERCIAL_LICENCES:
        return licence, None

    # Extract the article body text.
    body = root.find(".//body")
    if body is None:
        # Abstract-only article -- not enough content for a knowledge base.
        return licence, None

    # Flatten all text nodes, strip whitespace, join with spaces.
    text = " ".join(t.strip() for t in body.itertext() if t.strip())

    # Reject articles under 1,500 characters -- they are too short to be useful
    # as knowledge base chunks and would waste ingestion budget.
    return (licence, text) if len(text) > 1500 else (licence, None)


def main():
    """Main loop: search, filter, and save the corpus."""
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Searching PMC for open-access {GENE} articles...")

    # Over-fetch by 1.8× because many articles will be filtered out.
    # The typical yield after licence and length filtering is about 55%.
    ids = search_pmc(int(TARGET * 1.8))
    print(f"  {len(ids)} candidate PMCIDs")

    kept, skipped = 0, 0
    for pmcid in ids:
        if kept >= TARGET:
            break

        licence, text = fetch_article(pmcid)
        time.sleep(DELAY)  # respect NCBI rate limits

        if text is None:
            skipped += 1
            continue

        # Write the article as plain text with a header comment.
        with open(f"{OUTDIR}/PMC{pmcid}.txt", "w", encoding="utf-8") as f:
            f.write(f"# PMC{pmcid}  licence: {licence}\n\n{text}")
        kept += 1

        if kept % 50 == 0:
            print(f"  kept {kept}  (skipped {skipped} non-commercial / empty)")

    print(f"\nDone. {kept} commercial-use articles in ./{OUTDIR}/")
    print(f"Next:  aws s3 sync ./{OUTDIR} s3://YOUR-BUCKET/corpus/")


if __name__ == "__main__":
    main()
