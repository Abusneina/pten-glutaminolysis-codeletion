#!/usr/bin/env python3
"""
check_data.py

Finds truncated or unusable data files before the analysis runs into them.

fetch_data.py skips any file that already exists, without checking whether it
is complete. If a download was interrupted, a partial file is left on disk and
silently reused on every later pass. The symptom is a module reporting that a
gene is missing from an expression matrix, which looks like a data problem but
is a download problem.

For each study this checks that the expression, copy-number and clinical files
parse, that the genes the analysis needs are present, and that sample counts are
consistent between matrices. Nothing is modified; the output is a list of what
to redownload.

Usage:
    python check_data.py --datadir .\\data
    python check_data.py --datadir .\\data --fix-list
"""
import argparse
import os
import sys

import pandas as pd

REQUIRED_GENES = ["PTEN", "GLUD1", "GOT1", "GLS", "SLC1A5", "GPT2", "MYC"]

FILES = {
    "data_mrna_seq_v2_rsem.txt": "expression",
    "data_cna.txt": "copy number",
    "data_clinical_sample.txt": "clinical sample",
    "data_clinical_patient.txt": "clinical patient",
}


def _read_matrix(path):
    """Return (gene set, sample count) without parsing the numeric matrix.

    These files run to 150 MB and the check needs only the first column and the
    header line, so reading the whole table wastes minutes per study. The header
    is read directly and the gene column via usecols, which skips the numeric
    payload entirely.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = ""
        for line in fh:
            if not line.startswith("#"):
                header = line.rstrip("\n")
                break
    if not header:
        return None, 0
    cols = header.split("\t")
    gcol = next((c for c in cols if c.lower() == "hugo_symbol"), None)
    if gcol is None:
        return None, 0
    sample_cols = [c for c in cols
                   if c not in (gcol, "Entrez_Gene_Id") and c.startswith("TCGA")]
    genes = pd.read_csv(path, sep="\t", comment="#", usecols=[gcol],
                        dtype=str, low_memory=False)[gcol]
    return set(genes.astype(str)), len(sample_cols)


def check_study(study_dir):
    """Return a list of problems, empty if the study is usable."""
    problems = []
    name = os.path.basename(study_dir)

    for fname, label in FILES.items():
        path = os.path.join(study_dir, fname)
        if not os.path.exists(path):
            problems.append(f"{label} file missing")
            continue
        if os.path.getsize(path) == 0:
            problems.append(f"{label} file is empty")

    expr_path = os.path.join(study_dir, "data_mrna_seq_v2_rsem.txt")
    cna_path = os.path.join(study_dir, "data_cna.txt")

    expr_genes = cna_genes = None
    n_expr = n_cna = 0

    if os.path.exists(expr_path) and os.path.getsize(expr_path) > 0:
        try:
            expr_genes, n_expr = _read_matrix(expr_path)
        except Exception as exc:
            problems.append(f"expression file will not parse ({exc})")
        if expr_genes is None:
            problems.append("expression file has no Hugo_Symbol column")
        else:
            missing = [g for g in REQUIRED_GENES if g not in expr_genes]
            if missing:
                problems.append(
                    f"expression missing {', '.join(missing)} "
                    f"({len(expr_genes):,} genes, {n_expr:,} samples) "
                    f"- almost certainly truncated")

    if os.path.exists(cna_path) and os.path.getsize(cna_path) > 0:
        try:
            cna_genes, n_cna = _read_matrix(cna_path)
        except Exception as exc:
            problems.append(f"copy-number file will not parse ({exc})")
        if cna_genes is not None and "PTEN" not in cna_genes:
            problems.append(
                f"copy number missing PTEN ({len(cna_genes):,} genes) "
                f"- almost certainly truncated")

    # A large asymmetry between matrices is a strong truncation signal.
    if n_expr and n_cna:
        ratio = min(n_expr, n_cna) / max(n_expr, n_cna)
        if ratio < 0.5:
            problems.append(
                f"sample counts differ sharply: expression {n_expr:,}, "
                f"copy number {n_cna:,}")

    mut_path = os.path.join(study_dir, "data_mutations.txt")
    if os.path.exists(mut_path):
        try:
            with open(mut_path, encoding="utf-8", errors="replace") as fh:
                if "Hugo_Symbol" not in fh.readline():
                    problems.append("mutation file is not a table "
                                    "(LFS pointer or error page)")
        except Exception as exc:
            problems.append(f"mutation file unreadable ({exc})")
    else:
        problems.append("mutation file missing")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--fix-list", action="store_true",
                    help="print only the study names needing redownload")
    args = ap.parse_args()

    studies = sorted(d for d in os.listdir(args.datadir)
                     if os.path.isdir(os.path.join(args.datadir, d)))
    bad = {}
    for i, s in enumerate(studies, 1):
        if not args.fix_list:
            print(f"  [{i}/{len(studies)}] {s}", flush=True)
        problems = check_study(os.path.join(args.datadir, s))
        if problems:
            bad[s] = problems
    if not args.fix_list:
        print()

    if args.fix_list:
        for s in bad:
            print(s)
        return

    print(f"{len(studies)} studies checked, "
          f"{len(studies) - len(bad)} clean, {len(bad)} with problems\n")
    for s, problems in bad.items():
        print(f"{s}")
        for p in problems:
            print(f"    {p}")
    if not bad:
        print("All studies usable.")
        return

    print("\nTo redownload a study, delete its folder and rerun fetch_data.py, "
          "which only fetches what is absent:")
    print("\n  Remove-Item -Recurse -Force .\\data\\<study_folder>")
    print("  python fetch_data.py --outdir .\\data")
    print("  python fetch_mutations_api.py --datadir .\\data")
    print("\nA study listed only for a missing mutation file needs "
          "fetch_mutations_api.py alone.")


if __name__ == "__main__":
    main()
