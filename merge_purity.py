#!/usr/bin/env python3
"""
merge_purity.py

Adds tumor purity to a cohort's clinical sample file so prad_adjust.py can use
it as a covariate.

Why this matters. Prostate carcinoma is now the most interesting lineage in the
paper: its off-chromosome-10 genes decline, the decline is not explained by
copy number, and adjusting for grade, ERG and proliferation removes only 13
percent of it. The obvious remaining explanation is tumor composition. Bulk
RNA-seq measures a mixture of tumor, immune and stromal cells, and PTEN-deleted
prostate tumors differ from intact ones in cellularity. Without purity in the
model, that explanation cannot be excluded, and a reviewer will say so.

Two modes.

  --list-columns   prints every clinical column in a study, so you can see
                   whether purity is already present under a name the automatic
                   detection missed. Try this first; it costs nothing.

  --purity-file    merges an external purity table into data_clinical_sample.txt
                   as a TUMOR_PURITY column. prad_adjust.py then picks it up
                   automatically with no further change.

Where to get the purity table. The TCGA PanCanAtlas consensus purity estimates
come from the ABSOLUTE method and are published as a supplementary table on the
GDC PanCanAtlas publications page:

    https://gdc.cancer.gov/about-data/publications/pancanatlas

Look for the ABSOLUTE purity and ploidy file. It is a tab-delimited table with
one row per sample, carrying a sample barcode column and a purity column. This
script accepts it as-is and works out which columns those are.

Usage:
    python merge_purity.py --datadir .\\data --study prad --list-columns
    python merge_purity.py --datadir .\\data --study prad --purity-file purity.txt
    python merge_purity.py --datadir .\\data --purity-file purity.txt    (all studies)
"""
import argparse
import os
import shutil

import pandas as pd

SAMPLE_COL_CANDIDATES = ["SAMPLE", "ARRAY", "TUMOR_SAMPLE_BARCODE", "SAMPLE_ID",
                         "BARCODE", "ALIQUOT_BARCODE", "SAMPLE_BARCODE"]
PURITY_COL_CANDIDATES = ["PURITY", "TUMOR_PURITY", "ABSOLUTE_PURITY",
                         "CONSENSUS_PURITY", "PURITY_ABSOLUTE"]


def _norm(c):
    return str(c).strip().upper().replace(" ", "_").lstrip("#")


def list_columns(study_dir):
    for fname in ("data_clinical_sample.txt", "data_clinical_patient.txt"):
        path = os.path.join(study_dir, fname)
        if not os.path.exists(path):
            print(f"  {fname}: not present")
            continue
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False, nrows=5)
        cols = [_norm(c) for c in df.columns]
        print(f"\n  {fname}  ({len(cols)} columns)")
        for c in sorted(cols):
            mark = "  <-- looks like purity" if any(
                k in c for k in ("PURIT", "CELLULARIT", "ABSOLUTE")) else ""
            print(f"      {c}{mark}")


def find_cols(df):
    cols = {_norm(c): c for c in df.columns}
    sample = next((cols[k] for k in SAMPLE_COL_CANDIDATES if k in cols), None)
    if sample is None:
        sample = next((cols[k] for k in cols if "SAMPLE" in k or "BARCODE" in k), None)
    purity = next((cols[k] for k in PURITY_COL_CANDIDATES if k in cols), None)
    if purity is None:
        purity = next((cols[k] for k in cols if "PURIT" in k), None)
    return sample, purity


def merge(study_dir, purity_map, label):
    path = os.path.join(study_dir, "data_clinical_sample.txt")
    if not os.path.exists(path):
        print(f"  {label}: no clinical sample file, skipped")
        return False

    # cBioPortal clinical files carry four comment/header lines before the
    # real header. They must be preserved exactly or the loaders break.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    header_idx = next((i for i, l in enumerate(lines)
                       if not l.startswith("#")), 0)
    preamble = lines[:header_idx]

    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    sid = next((c for c in df.columns if _norm(c) == "SAMPLE_ID"), None)
    if sid is None:
        print(f"  {label}: no SAMPLE_ID column, skipped")
        return False

    keys = df[sid].astype(str).str[:15]
    values = keys.map(purity_map)
    matched = int(values.notna().sum())
    if matched == 0:
        print(f"  {label}: no samples matched the purity table, skipped")
        return False

    existing = next((c for c in df.columns if _norm(c) == "TUMOR_PURITY"), None)
    if existing:
        df[existing] = values
    else:
        df["TUMOR_PURITY"] = values

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)

    body = df.to_csv(sep="\t", index=False)
    # Extend each preamble line so the column count still matches the header.
    new_preamble = []
    for l in preamble:
        parts = l.rstrip("\n").split("\t")
        parts.append("TUMOR_PURITY" if not l.startswith("#1") else "1")
        new_preamble.append("\t".join(parts) + "\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(new_preamble)
        fh.write(body)

    pct = 100 * matched / len(df)
    print(f"  {label}: purity merged for {matched} of {len(df)} samples "
          f"({pct:.0f}%); original saved as data_clinical_sample.txt.bak")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--study", default=None,
                    help="study folder prefix, e.g. prad; default is all")
    ap.add_argument("--list-columns", action="store_true")
    ap.add_argument("--purity-file", default=None)
    args = ap.parse_args()

    studies = sorted(d for d in os.listdir(args.datadir)
                     if os.path.isdir(os.path.join(args.datadir, d))
                     and (args.study is None or d.startswith(args.study)))
    if not studies:
        raise SystemExit(f"No study folders matched in {args.datadir}")

    if args.list_columns:
        for s in studies:
            print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")
            list_columns(os.path.join(args.datadir, s))
        return

    if not args.purity_file:
        raise SystemExit("Give either --list-columns or --purity-file")

    # Sniff the delimiter, since these tables ship as tab or comma delimited.
    pur = pd.read_csv(args.purity_file, sep=None, engine="python")
    sample_col, purity_col = find_cols(pur)
    if sample_col is None or purity_col is None:
        print("Could not identify the sample and purity columns. Found:")
        for c in pur.columns:
            print(f"    {c}")
        raise SystemExit("Rename them, or tell me the column names.")

    print(f"Purity table: {len(pur):,} rows; "
          f"sample column '{sample_col}', purity column '{purity_col}'")
    pur["_key"] = pur[sample_col].astype(str).str[:15]
    pur["_val"] = pd.to_numeric(pur[purity_col], errors="coerce")
    purity_map = pur.dropna(subset=["_val"]).drop_duplicates("_key") \
                    .set_index("_key")["_val"].to_dict()
    print(f"{len(purity_map):,} samples with a usable purity value\n")

    done = 0
    for i, s in enumerate(studies, 1):
        print(f"  [{i}/{len(studies)}] {s}", flush=True)
        if merge(os.path.join(args.datadir, s), purity_map, s):
            done += 1

    print(f"\n{done} of {len(studies)} studies updated.")
    if done:
        print("\nNow rerun the prostate adjustment:")
        print("    python -u prad_adjust.py --datadir .\\data --outdir .\\results")
        print("\nThe covariate block should now list purity with its source "
              "rather than NOT AVAILABLE.")


if __name__ == "__main__":
    main()
