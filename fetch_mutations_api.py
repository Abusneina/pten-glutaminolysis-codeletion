#!/usr/bin/env python3
"""
fetch_mutations_api.py

Fallback for mutation files that will not download from the cBioPortal datahub.

The datahub serves data_mutations.txt through Git LFS. When the repository
exceeds its LFS bandwidth quota, GitHub returns an error object instead of a
download link and the fetch fails with a KeyError on 'actions'. Retrying does
not help, because the quota resets on GitHub's schedule and not ours.

The analysis only ever reads PTEN mutations. The full MAF files are 12 to 75 MB
each and carry every gene; what is actually needed is a few hundred rows. This
script queries the cBioPortal REST API for PTEN mutations in one study and
writes a data_mutations.txt containing exactly those, with the same column names
and the same MAF variant vocabulary the datahub files use. The downstream
modules read it without modification.

Provenance is unchanged: this is the same cBioPortal study data, retrieved
through the public API rather than the datahub mirror. The Methods should say
so, since a reader reproducing the work needs to know which route was used.

Only studies with a missing or empty data_mutations.txt are touched. Existing
files are left alone, so this can be run after fetch_data.py without risk.

Usage:
    python fetch_mutations_api.py --datadir .\\data
    python fetch_mutations_api.py --datadir .\\data --studies brca_tcga_pan_can_atlas_2018
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

API = "https://www.cbioportal.org/api"
PTEN_ENTREZ = 5728

COLUMNS = ["Hugo_Symbol", "Entrez_Gene_Id", "Tumor_Sample_Barcode",
           "Variant_Classification", "Variant_Type", "HGVSp_Short",
           "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]


def _get(url, timeout=60, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json",
                              "User-Agent": "pten-codeletion-analysis"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except Exception as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise last


def fetch_study(study_id):
    """Return a list of PTEN mutation records, or None if unavailable."""
    profile = f"{study_id}_mutations"
    sample_list = f"{study_id}_all"
    url = (f"{API}/molecular-profiles/{profile}/mutations"
           f"?sampleListId={sample_list}"
           f"&entrezGeneId={PTEN_ENTREZ}"
           f"&projection=DETAILED")
    return _get(url)


def write_maf(records, path):
    """Write the records in datahub MAF column format."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in records:
            gene = (r.get("gene") or {})
            row = [
                gene.get("hugoGeneSymbol", "PTEN"),
                str(r.get("entrezGeneId", PTEN_ENTREZ)),
                r.get("sampleId", ""),
                r.get("mutationType", ""),
                r.get("variantType", ""),
                r.get("proteinChange", ""),
                str(r.get("startPosition", "")),
                r.get("referenceAllele", ""),
                r.get("variantAllele", ""),
            ]
            fh.write("\t".join(str(x) for x in row) + "\n")


def needs_fetch(study_dir):
    """True when the mutation file is absent, or is not a usable table.

    Size is not a reliable test: a PTEN-only file for a small cohort is legitimately
    tiny, while a Git LFS pointer stub is also tiny. Check the content instead.
    A valid file has a Hugo_Symbol header and at least one data row.
    """
    path = os.path.join(study_dir, "data_mutations.txt")
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
            if "Hugo_Symbol" not in header:
                return True          # LFS pointer, error page, or truncated
            for line in fh:
                if line.strip():
                    return False     # header plus at least one record
        return True                  # header only
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--studies", nargs="*", default=None,
                    help="specific study folder names; default is all missing")
    ap.add_argument("--force", action="store_true",
                    help="refetch even where a mutation file already exists")
    args = ap.parse_args()

    if not os.path.isdir(args.datadir):
        raise SystemExit(f"{args.datadir} not found")

    studies = args.studies or sorted(
        d for d in os.listdir(args.datadir)
        if os.path.isdir(os.path.join(args.datadir, d)))

    fetched = skipped = failed = 0
    for s in studies:
        sdir = os.path.join(args.datadir, s)
        if not os.path.isdir(sdir):
            print(f"  [miss] {s}: no such folder")
            continue
        if not args.force and not needs_fetch(sdir):
            skipped += 1
            continue

        try:
            records = fetch_study(s)
        except Exception as exc:
            print(f"  [FAIL] {s}: {exc}")
            failed += 1
            continue

        if records is None:
            print(f"  [none] {s}: no mutation profile at the API")
            failed += 1
            continue

        out = os.path.join(sdir, "data_mutations.txt")
        write_maf(records, out)
        size = os.path.getsize(out)
        n_samples = len({r.get("sampleId") for r in records})
        print(f"  [ok]   {s}: {len(records)} PTEN mutations in "
              f"{n_samples} samples ({size:,} bytes)")
        fetched += 1
        time.sleep(0.5)

    print(f"\n{fetched} fetched, {skipped} already present, {failed} failed")
    if fetched:
        print("\nThese files contain PTEN mutations only, which is all the "
              "analysis reads. Note in the Methods that mutation data were "
              "retrieved through the cBioPortal REST API.")
    if failed:
        print("\nFor any study still failing, download the full study archive "
              "from https://www.cbioportal.org/datasets and place its "
              "data_mutations.txt in the study folder.")


if __name__ == "__main__":
    main()
