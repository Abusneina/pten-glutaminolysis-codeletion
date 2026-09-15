#!/usr/bin/env python3
"""
msi_stratification.py

Closes Reviewer 1, point 2.

Endometrial carcinoma dominates the mutation analysis, contributing 320 of the
mutant copy-neutral tumors, and it is also the cohort in which the natural
experiment gives a different answer. The reviewer asks whether the mutation
filtering accounted for microsatellite instability, on the reasoning that a
hypermutated background generates passenger mutations in metabolic genes and
could produce an apparent effect without a mechanism.

This repeats the natural experiment with tumors stratified by MSI status and by
tumor mutational burden, so the endometrial exception can be characterised
rather than simply flagged.

The question it answers is specific: does PTEN mutation alone still lower GLUD1
in endometrial carcinoma once hypermutated tumors are removed? If it does, the
exception is real and the MSI explanation is dismissed on evidence. If it does
not, the exception is an artifact of mutational burden, which would be a cleaner
result than the one currently in the manuscript.

MSI status is looked for under several column names, since cBioPortal is not
consistent across studies. Whatever is found is named in the output. If nothing
is found, the module says so rather than silently proceeding, and TMB is used as
a fallback stratifier.

Usage:
    python msi_stratification.py --datadir ./data --outdir ./results
    python msi_stratification.py --datadir ./data --outdir ./results --cohort UCEC
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A
from mutation_experiment import load_mutations, _contrast

MSI_COLS = ["MSI_STATUS", "MSI_SENSOR_SCORE", "MSI_SCORE_MANTIS",
            "MSI_TYPE", "MSI_STATUS_MANTIS"]
TMB_COLS = ["TMB_NONSYNONYMOUS", "MUTATION_COUNT", "TMB"]

# Conventional thresholds. Both are reported so a reader can see which was used.
MANTIS_MSI_H = 0.4
SENSOR_MSI_H = 10.0
TMB_HIGH = 10.0

GENE = "GLUD1"
MARGIN = 0.25


def _norm(c):
    return str(c).strip().upper().replace(" ", "_").lstrip("#")


def _clinical(study_dir, fname):
    path = os.path.join(study_dir, fname)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = [_norm(c) for c in df.columns]
    return df


def classify_msi(stab, ptab, samples):
    """Return (series of 'MSI-H'/'MSS'/nan, description of the source)."""
    for tbl, key, kind in ((stab, "SAMPLE_ID", "sample"), (ptab, "PATIENT_ID", "patient")):
        if tbl.empty or key not in tbl.columns:
            continue
        t = tbl.drop_duplicates(key).copy()
        t[key] = t[key].astype(str)
        if kind == "sample":
            t[key] = t[key].str[:15]
            index = pd.Index(samples)
        else:
            index = pd.Index([s[:12] for s in samples])
        t = t.set_index(key)

        for col in MSI_COLS:
            if col not in t.columns:
                continue
            raw = t[col].reindex(index)
            if raw.notna().sum() < 10:
                continue
            num = pd.to_numeric(raw, errors="coerce")
            if num.notna().sum() >= 10:
                thr = MANTIS_MSI_H if "MANTIS" in col else SENSOR_MSI_H
                out = pd.Series(np.where(num >= thr, "MSI-H", "MSS"),
                                index=range(len(samples)))
                out[num.isna().values] = np.nan
                return out, f"{kind} column {col}, threshold {thr}"
            txt = raw.astype(str).str.upper()
            out = pd.Series(np.where(txt.str.contains("MSI-H|INSTABLE|HIGH"),
                                     "MSI-H",
                                     np.where(txt.str.contains("MSS|STABLE|LOW"),
                                              "MSS", np.nan)),
                            index=range(len(samples)))
            if (out != "nan").sum() >= 10:
                return out, f"{kind} column {col}, categorical"
    return pd.Series([np.nan] * len(samples)), "NOT AVAILABLE"


def classify_tmb(stab, ptab, samples):
    for tbl, key, kind in ((stab, "SAMPLE_ID", "sample"), (ptab, "PATIENT_ID", "patient")):
        if tbl.empty or key not in tbl.columns:
            continue
        t = tbl.drop_duplicates(key).copy()
        t[key] = t[key].astype(str)
        index = pd.Index(samples) if kind == "sample" else pd.Index([s[:12] for s in samples])
        if kind == "sample":
            t[key] = t[key].str[:15]
        t = t.set_index(key)
        for col in TMB_COLS:
            if col in t.columns:
                num = pd.to_numeric(t[col].reindex(index), errors="coerce")
                if num.notna().sum() >= 10:
                    return num.values, f"{kind} column {col}"
    return np.full(len(samples), np.nan), "NOT AVAILABLE"


def run(datadir, outdir, cohort=None, margin=MARGIN):
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for d in sorted(os.listdir(datadir)):
        sdir = os.path.join(datadir, d)
        if not os.path.isdir(sdir):
            continue
        label = A.STUDY_LABELS.get(d, d)
        if cohort and label != cohort:
            continue
        try:
            expr = A.load_expression(sdir)
            cna = A.load_cna(sdir)
        except Exception:
            continue
        tiers = A.call_pten_dosage(expr, cna)
        if tiers is None or not len(tiers) or GENE not in expr.index:
            continue
        counts = tiers.value_counts()
        if (counts.get("HemDel", 0) < A.MIN_TIER or counts.get("HomDel", 0) < A.MIN_TIER
                or counts.get("Intact", 0) < A.MIN_INTACT):
            continue

        mutants = load_mutations(sdir, "PTEN", truncating_only=False)
        if mutants is None:
            print(f"  [skip] {label}: no mutation file")
            continue

        tiers = tiers.dropna()
        cols = list(tiers.index)
        stab, ptab = _clinical(sdir, "data_clinical_sample.txt"), \
            _clinical(sdir, "data_clinical_patient.txt")
        msi, msi_src = classify_msi(stab, ptab, cols)
        tmb, tmb_src = classify_tmb(stab, ptab, cols)

        print(f"\n{label}: MSI from {msi_src}; TMB from {tmb_src}")

        vals = np.log2(expr.loc[GENE, cols].astype(float).clip(lower=0) + 1)
        z = ((vals - vals.mean()) / vals.std(ddof=0)).values
        is_mut = np.array([c in mutants for c in cols])
        tier_arr = tiers.values
        msi_arr = np.asarray(msi, dtype=object)

        strata = {"all": np.ones(len(cols), bool)}
        if (msi_arr == "MSS").sum() >= 10:
            strata["MSS only"] = msi_arr == "MSS"
        if (msi_arr == "MSI-H").sum() >= 10:
            strata["MSI-H only"] = msi_arr == "MSI-H"
        if np.isfinite(tmb).sum() >= 20:
            strata["TMB below threshold"] = np.nan_to_num(tmb, nan=np.inf) < TMB_HIGH

        for sname, mask in strata.items():
            mut_neutral = z[mask & (tier_arr == "Intact") & is_mut]
            wt_neutral = z[mask & (tier_arr == "Intact") & ~is_mut]
            hem = z[mask & (tier_arr == "HemDel")]
            row = {"study": label, "stratum": sname, "gene": GENE,
                   "msi_source": msi_src, "tmb_source": tmb_src}
            row.update(_contrast("mut_vs_wt", mut_neutral, wt_neutral, margin))
            row.update(_contrast("hem_vs_wt", hem, wt_neutral, margin))
            rows.append(row)
            if row.get("p_mut_vs_wt") is not None:
                print(f"  {sname}: mutant n={row['n_mut_vs_wt_test']} vs "
                      f"wild-type n={row['n_mut_vs_wt_ref']}; "
                      f"diff {row['diff_mut_vs_wt']:+.3f}; "
                      f"P={row['p_mut_vs_wt']:.3g}")
            else:
                print(f"  {sname}: too few tumors for a contrast")

    res = pd.DataFrame(rows)
    if res.empty:
        print("[msi_stratification] nothing to report")
        return res
    out = os.path.join(outdir, "msi_stratified_mutation_experiment.csv")
    res.to_csv(out, index=False)
    print(f"\n[msi_stratification] {len(res)} rows -> {out}")
    print("  Read the MSS-only row for endometrial carcinoma. If the mutation "
          "effect survives there, microsatellite instability is not the "
          "explanation and the exception stands.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--cohort", default=None)
    args = ap.parse_args()
    run(args.datadir, args.outdir, args.cohort)


if __name__ == "__main__":
    main()
