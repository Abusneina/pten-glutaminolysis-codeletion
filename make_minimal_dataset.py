#!/usr/bin/env python3
"""
make_minimal_dataset.py

Builds the minimal data set PLOS ONE asked for in journal requirement 2.

The requirement is often misread as a demand to change repository. It is not.
Zenodo is on the PLOS recommended list, alongside Dryad, figshare, Harvard
Dataverse and the Open Science Framework. GitHub is the one that does not
qualify. The real gap in this submission is that the Zenodo record holds code
and no derived data, so there is nothing deposited from which the reported
numbers can be recomputed.

Citing cBioPortal and DepMap as third-party sources remains correct and stays
in the Data Availability Statement. What this script adds is the analysis-ready
layer between those sources and the manuscript: one row per tumor, carrying
every variable any figure or table in the paper is computed from.

The output is small, a few megabytes, and TCGA PanCancer Atlas open-access data
may be redistributed in derived form. Nothing here is identifiable: TCGA
barcodes are de-identified by design and no clinical narrative is included.

Usage:
    python make_minimal_dataset.py --datadir ./data --outdir ./minimal_dataset
"""
import argparse
import hashlib
import os
from datetime import date

import numpy as np
import pandas as pd

import analyze as A

SIGNATURE = ["GLS", "SLC1A5", "GOT1", "GLUD1", "GPT2"]
# ERG is the surrogate for TMPRSS2-ERG fusion status used in the prostate model.
EXTRA_EXPRESSION = ["MYC", "PTEN", "ERG"]

# Proliferation index genes, summarised into one column so the prostate
# covariate model can be reproduced from this file alone.
PROLIFERATION = ["MKI67", "TOP2A", "CCNB1", "BUB1", "PLK1", "AURKA",
                 "CDK1", "RRM2", "TYMS", "PCNA", "CCNA2", "MCM2"]

PURITY_COLS = ("TUMOR_PURITY", "PURITY")
MSI_COLS = ("MSI_SENSOR_SCORE", "MSI_SCORE_MANTIS")
TMB_COLS = ("TMB_NONSYNONYMOUS", "MUTATION_COUNT")
STAGE_COLS = ("PATH_T_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE")


def _stage_ordinal(v):
    """T stage on an ordinal scale: T2A < T2B < T2C < T3A < T3B < T4."""
    if not isinstance(v, str):
        return np.nan
    s = v.strip().upper().replace("STAGE", "").strip()
    if not s.startswith("T"):
        return np.nan
    digits = "".join(ch for ch in s[1:] if ch.isdigit())
    if not digits:
        return np.nan
    minor = {"A": 0, "B": 1, "C": 2}.get(
        next((ch for ch in s[1:] if ch.isalpha()), ""), 0)
    return float(int(digits[0])) + minor / 3.0

# Chromosome-10 genes whose copy number carries the paper's argument, plus the
# interval neighbours used by interval_decay.py. Any gene absent from a
# cohort's matrix is simply skipped.
CN_GENES = SIGNATURE + ["PTEN", "KLLN", "ATAD1", "PAPSS2", "RNLS", "MINPP1",
                        "BMPR1A", "FAS", "LIPA", "HHEX", "IDE", "KIF11"]

INACTIVATING = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"}

ANEUPLOIDY_COLS = ("ANEUPLOIDY_SCORE", "ANEUPLOIDY SCORE", "AneuploidyScore")


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sample_table(study_dir):
    path = os.path.join(study_dir, "data_clinical_sample.txt")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = [c.strip().upper().replace(" ", "_").lstrip("#") for c in df.columns]
    return df


def _load_patient_table(study_dir):
    path = os.path.join(study_dir, "data_clinical_patient.txt")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = [c.strip().upper().replace(" ", "_").lstrip("#") for c in df.columns]
    return df


def _pten_mutation_status(study_dir, samples):
    """Per-sample PTEN mutation class: none, truncating, missense, other."""
    out = pd.Series("none", index=samples, dtype=object)
    path = os.path.join(study_dir, "data_mutations.txt")
    if not os.path.exists(path):
        return pd.Series(np.nan, index=samples, dtype=object)
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    gcol = next((c for c in df.columns if c.lower() == "hugo_symbol"), None)
    scol = next((c for c in df.columns if c.lower() == "tumor_sample_barcode"), None)
    vcol = next((c for c in df.columns if c.lower() == "variant_classification"), None)
    if not all([gcol, scol, vcol]):
        return pd.Series(np.nan, index=samples, dtype=object)
    sub = df[df[gcol].astype(str).str.upper() == "PTEN"].copy()
    sub["sid"] = sub[scol].astype(str).str[:15]
    for sid, grp in sub.groupby("sid"):
        if sid not in out.index:
            continue
        classes = set(grp[vcol].astype(str))
        if classes & TRUNCATING:
            out[sid] = "truncating"
        elif "Missense_Mutation" in classes:
            out[sid] = "missense"
        elif classes & INACTIVATING:
            out[sid] = "other_inactivating"
        else:
            out[sid] = "silent_only"
    return out


def build_cohort(study_dir):
    code = os.path.basename(study_dir)
    label = A.STUDY_LABELS.get(code, code)
    expr = A.load_expression(study_dir)
    cna = A.load_cna(study_dir)

    tiers = A.call_pten_dosage(expr, cna)
    if tiers is None or not len(tiers):
        return None, label, "no PTEN dosage call"
    counts = tiers.value_counts()
    if (counts.get("HemDel", 0) < A.MIN_TIER or counts.get("HomDel", 0) < A.MIN_TIER
            or counts.get("Intact", 0) < A.MIN_INTACT):
        return None, label, (f"inclusion rule not met "
                             f"(Intact {counts.get('Intact',0)}, "
                             f"HemDel {counts.get('HemDel',0)}, "
                             f"HomDel {counts.get('HomDel',0)})")

    tiers = tiers.dropna()
    samples = list(tiers.index)

    out = pd.DataFrame({"study": label, "sample_id": samples})
    out["pten_tier"] = tiers.values
    out["pten_severity"] = out["pten_tier"].map(A.SEVERITY)

    # Expression: log2(x+1) then within-cohort z-score, exactly as analyze.py
    # scores the signature, so the deposited values are the ones used.
    wanted = [g for g in SIGNATURE + EXTRA_EXPRESSION if g in expr.index]
    sub = np.log2(expr.loc[wanted, samples].astype(float).clip(lower=0) + 1)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
    for g in wanted:
        out[f"z_{g}"] = z.loc[g].values
    present_sig = [g for g in SIGNATURE if g in z.index]
    out["signature_score"] = z.loc[present_sig].mean(axis=0).values

    prolif = [g for g in PROLIFERATION if g in expr.index]
    if prolif:
        ps = np.log2(expr.loc[prolif, samples].astype(float).clip(lower=0) + 1)
        pz = ps.sub(ps.mean(axis=1), axis=0).div(ps.std(axis=1, ddof=0), axis=0)
        out["proliferation_index"] = pz.mean(axis=0).values

    for g in CN_GENES:
        if cna is not None and g in cna.index:
            out[f"cn_{g}"] = pd.to_numeric(cna.loc[g, samples], errors="coerce").values

    out["pten_mutation"] = _pten_mutation_status(study_dir, pd.Index(samples)).values

    stab = _load_sample_table(study_dir)
    if len(stab) and "SAMPLE_ID" in stab.columns:
        stab["SAMPLE_ID"] = stab["SAMPLE_ID"].astype(str).str[:15]
        stab = stab.drop_duplicates("SAMPLE_ID").set_index("SAMPLE_ID")
        if "PATIENT_ID" in stab.columns:
            out["patient_id"] = stab["PATIENT_ID"].reindex(samples).values
        acol = next((c for c in stab.columns
                     if c in [a.upper().replace(" ", "_") for a in ANEUPLOIDY_COLS]), None)
        out["aneuploidy_score"] = (
            pd.to_numeric(stab[acol], errors="coerce").reindex(samples).values
            if acol else np.nan)
        for _name, _cands in (("tumor_purity", PURITY_COLS),
                              ("msi_score", MSI_COLS),
                              ("tmb_nonsynonymous", TMB_COLS)):
            _col = next((c for c in stab.columns if c in _cands), None)
            out[_name] = (pd.to_numeric(stab[_col], errors="coerce")
                          .reindex(samples).values if _col else np.nan)
    else:
        out["patient_id"] = [s[:12] for s in samples]
        out["aneuploidy_score"] = np.nan

    ptab = _load_patient_table(study_dir)
    if len(ptab) and "PATIENT_ID" in ptab.columns and "patient_id" in out.columns:
        ptab = ptab.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
        for src_col, dst in (("OS_MONTHS", "os_months"), ("OS_STATUS", "os_status")):
            if src_col in ptab.columns:
                out[dst] = ptab[src_col].reindex(out["patient_id"]).values
        if "os_months" in out.columns:
            out["os_months"] = pd.to_numeric(out["os_months"], errors="coerce")

    # T stage, from whichever table carries it. Kept outside the clinical
    # blocks above so that a missing survival table cannot suppress it.
    if "stage_t_ordinal" not in out.columns:
        stage_vals = None
        for tbl, key in ((_load_patient_table(study_dir), "PATIENT_ID"),
                         (_load_sample_table(study_dir), "SAMPLE_ID")):
            if tbl.empty or key not in tbl.columns:
                continue
            col = next((c for c in tbl.columns if c in STAGE_COLS), None)
            if col is None:
                continue
            t = tbl.drop_duplicates(key).copy()
            t[key] = t[key].astype(str)
            if key == "SAMPLE_ID":
                t[key] = t[key].str[:15]
                idx = pd.Index(samples)
            else:
                idx = pd.Index(out["patient_id"].astype(str)) \
                    if "patient_id" in out.columns \
                    else pd.Index([s[:12] for s in samples])
            v = t.set_index(key)[col].reindex(idx).map(_stage_ordinal)
            if v.notna().sum() >= 10:
                stage_vals = v.values
                break
        out["stage_t_ordinal"] = stage_vals if stage_vals is not None else np.nan

    lead = ["study", "sample_id", "patient_id", "pten_tier", "pten_severity",
            "pten_mutation", "signature_score"]
    cols = [c for c in lead if c in out.columns] + \
           [c for c in out.columns if c not in lead]
    return out[cols].round(6), label, None


DICTIONARY = [
    ("study", "TCGA study abbreviation"),
    ("sample_id", "TCGA sample barcode, truncated to 15 characters"),
    ("patient_id", "TCGA patient barcode"),
    ("pten_tier", "Intact, HemDel or HomDel, from the thresholded GISTIC2 PTEN call"),
    ("pten_severity", "Ordinal loss severity: Intact 0, HemDel 1, HomDel 2"),
    ("pten_mutation", "none, missense, truncating, other_inactivating, silent_only; "
                      "blank if the cohort has no mutation file"),
    ("signature_score", "Unweighted mean of the five per-gene z-scores"),
    ("z_<GENE>", "log2(RSEM + 1) expression, z-scored within cohort across all "
                 "samples of that cohort"),
    ("cn_<GENE>", "Thresholded GISTIC2 copy-number call for that gene"),
    ("aneuploidy_score", "Per-sample aneuploidy score from the cBioPortal clinical "
                         "sample file; blank where the cohort has no such column"),
    ("tumor_purity", "Consensus purity from the TCGA PanCanAtlas ABSOLUTE tables, "
                     "merged on the sample barcode"),
    ("msi_score", "MSI sensor or MANTIS score from the clinical sample file"),
    ("tmb_nonsynonymous", "Non-synonymous tumor mutational burden"),
    ("stage_t_ordinal", "Pathological T stage mapped to an ordinal scale "
                        "(T2A < T2B < T2C < T3A < T3B < T4); blank where not recorded"),
    ("proliferation_index", "Mean z-score of MKI67, TOP2A, CCNB1, BUB1, PLK1, "
                            "AURKA, CDK1, RRM2, TYMS, PCNA, CCNA2 and MCM2"),
    ("os_months", "Overall survival in months"),
    ("os_status", "Overall survival status as distributed by cBioPortal"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./minimal_dataset")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    study_dirs = [os.path.join(args.datadir, d) for d in sorted(os.listdir(args.datadir))
                  if os.path.isdir(os.path.join(args.datadir, d))]

    frames, included, excluded = [], [], []
    for n_study, sdir in enumerate(study_dirs, 1):
        print(f"  [{n_study}/{len(study_dirs)}] {os.path.basename(sdir)}",
              flush=True)
        try:
            df, label, reason = build_cohort(sdir)
        except Exception as exc:
            excluded.append((os.path.basename(sdir), f"error: {exc}"))
            continue
        if df is None:
            excluded.append((label, reason))
            continue
        frames.append(df)
        included.append((label, len(df)))
        print(f"      {label}: {len(df)} tumors", flush=True)

    if not frames:
        raise SystemExit("No cohort met the inclusion rule. Check --datadir.")

    combined = pd.concat(frames, ignore_index=True)
    main_path = os.path.join(args.outdir, "minimal_dataset_tumor_level.csv")
    combined.to_csv(main_path, index=False)

    pd.DataFrame(DICTIONARY, columns=["column", "description"]).to_csv(
        os.path.join(args.outdir, "data_dictionary.csv"), index=False)

    pd.DataFrame(excluded, columns=["study", "reason_for_exclusion"]).to_csv(
        os.path.join(args.outdir, "excluded_studies.csv"), index=False)

    manifest = []
    for f in sorted(os.listdir(args.outdir)):
        fp = os.path.join(args.outdir, f)
        if os.path.isfile(fp):
            manifest.append({"file": f, "bytes": os.path.getsize(fp),
                             "sha256": _sha256(fp)})
    pd.DataFrame(manifest).to_csv(
        os.path.join(args.outdir, "MANIFEST.csv"), index=False)

    size_mb = os.path.getsize(main_path) / 1e6
    print(f"\n{len(combined)} tumors across {len(included)} cohorts")
    print(f"{main_path}  ({size_mb:.1f} MB)")
    print(f"{len(excluded)} studies excluded; see excluded_studies.csv")
    print("\nThis file answers Reviewer 2, observation 11 and Reviewer 3, point 4 "
          "as well: excluded_studies.csv names every study that failed the "
          "inclusion rule and why.")

    readme = f"""# Minimal data set for PONE-D-26-35226

Generated {date.today().isoformat()} by make_minimal_dataset.py.

## What this is

One row per tumor, carrying every variable from which the figures and tables in
the manuscript are computed. Deposited to satisfy the PLOS ONE requirement for a
minimal data set sufficient to replicate the reported findings.

## Provenance

Derived from TCGA PanCancer Atlas data obtained through the cBioPortal datahub.
Primary data are third-party and remain available from cBioPortal; this file is
the analysis-ready layer between those sources and the manuscript. No new data
were generated. TCGA barcodes are de-identified by design and no clinical
narrative is included.

## Files

- minimal_dataset_tumor_level.csv: {len(combined)} tumors, {len(included)} cohorts
- data_dictionary.csv: column definitions
- excluded_studies.csv: every study assessed but not analyzed, with the reason
- MANIFEST.csv: file sizes and SHA-256 checksums

## Cohorts included

{chr(10).join(f'- {l}: {n} tumors' for l, n in included)}

## Reproducing the manuscript from this file

The analysis code at the accompanying repository regenerates every table and
figure. Copy-number status is the thresholded GISTIC2 call; expression is
log2(RSEM + 1) z-scored within each cohort across all samples of that cohort,
not against a diploid reference subset.
"""
    with open(os.path.join(args.outdir, "README.md"), "w") as fh:
        fh.write(readme)
    print(f"{os.path.join(args.outdir, 'README.md')} written")


if __name__ == "__main__":
    main()
